"""
Agentic Engine — v1

What this prototype demonstrates
--------------------------------
• Agentic skeleton with distinct agents: PlannerAgent, TrendAgent, DiscoveryAgent, WriterAgent, StoreAgent
• Dynamic keyword expansion from Google Trends that feeds directly into the next cycle's plan
• Per-platform global caps (so the run is predictable for demos)
• Deduplication + cooldown to avoid re-commenting the same URL too soon
• Generation of empathetic, platform-aware copy (no links) via OpenAI
• Storage to SQLite + CSV for your Streamlit dashboard

This file is self-contained and can run without the previous bot file.
Posting to platforms is intentionally excluded in v1 (we'll add PublisherAgent in v2).

Quick start
-----------
1) pip install -r requirements.txt  (see below)
2) Create .env with:
   OPENAI_API_KEY=...
   YOUTUBE_API_KEY=...
   # Reddit (discovery only)
   REDDIT_CLIENT_ID=...
   REDDIT_CLIENT_SECRET=...
   REDDIT_USER_AGENT=careyaya-agent/0.1 by <your-username>
3) python agentic_engine.py --plan-and-run

Requirements.txt
----------------
openai
python-dotenv
pandas
requests
google-api-python-client
google-auth-oauthlib
google-auth-httplib2
praw
pytrends

CLI
---
--plan-only     : build plan.json from static + trends, exit
--plan-and-run  : build plan then run discovery + writing + store
--dry-run       : reuse existing plan.json, run discovery + writing + store

Output
------
output/plan.json               # chosen keywords + platform caps for this run
output/engine.db               # sqlite history (for dedupe/cooldown)
output/comments_posted.csv     # dashboard data source
output/logs/run_*.csv          # per-run detailed log
"""

import os
import sys
import json
import time
import pickle
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pickle

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads .env in this folder

# external APIs
from pytrends.request import TrendReq
from googleapiclient.discovery import build as gbuild
import praw

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------
# Constants / Defaults
# ----------------------
OUTPUT_DIR = "output"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PLAN_PATH = os.path.join(OUTPUT_DIR, "plan.json")
CSV_LOG = os.path.join(OUTPUT_DIR, "comments_posted.csv")
SQLITE_PATH = os.path.join(OUTPUT_DIR, "engine.db")

DEFAULT_KEYWORDS = [
    "elder care",
    "caregiver burnout",
    "home health care",
    "dementia caregiving",
    "AI caregiving",
]
STATIC_KEYWORD_SET = {kw.lower() for kw in DEFAULT_KEYWORDS}

# hard global caps for demo predictability
LIMIT_YOUTUBE_TOTAL = 2
LIMIT_REDDIT_TOTAL = 2
LIMIT_TRENDS_TOTAL = 5

COOLDOWN_DAYS = 7

# --------------
# Utilities
# --------------

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -------------------------
# Storage / SQLite helpers..
# -------------------------
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
  id INTEGER PRIMARY KEY,
  platform TEXT NOT NULL,
  keyword TEXT,
  topic TEXT,
  url TEXT,
  generated_text TEXT,
  posted INTEGER DEFAULT 0,
  posted_at TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_posts_url ON posts(url);
CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform);
"""


def db_connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def db_init(conn: sqlite3.Connection):
    with conn:
        conn.executescript(DB_SCHEMA)


def db_already_recent(conn: sqlite3.Connection, url: str, cooldown_days: int) -> bool:
    cutoff = (datetime.now() - timedelta(days=cooldown_days)).strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        "SELECT 1 FROM posts WHERE url=? AND created_at >= ? LIMIT 1",
        (url, cutoff),
    )
    return cur.fetchone() is not None



def db_insert_post(conn: sqlite3.Connection, row: Dict[str, Any]):
    with conn:
        conn.execute(
            """
            INSERT INTO posts(platform, keyword, topic, url, generated_text, posted, posted_at)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                row.get("platform"),
                row.get("keyword"),
                row.get("topic"),
                row.get("url"),
                row.get("generated_text"),
                1 if row.get("posted") else 0,
                row.get("posted_at"),
            ),
        )


# ----------------------
# Agent definitions
# ----------------------
@dataclass
class Plan:
    keywords: List[str]
    limits: Dict[str, int]   # e.g., {"youtube": 2, "reddit": 2, "trends": 5}


class TrendAgent:
    def __init__(self, geo: str = "US"):
        self.geo = geo
        self.cache_path = os.path.join(OUTPUT_DIR, "trends_cache.json")

    def _load_cache(self):
        """Return cached queries if under 24h old."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    cache = json.load(f)
                fetched = datetime.fromisoformat(cache.get("fetched_at", "1970-01-01 00:00:00"))
                if (datetime.now() - fetched) < timedelta(hours=24):
                    print("♻️ Using cached Google Trends data (<24h old).")
                    return cache.get("queries", [])
            except Exception:
                pass
        return []

    def _save_cache(self, queries: List[str]):
        try:
            with open(self.cache_path, "w") as f:
                json.dump({"fetched_at": now_iso(), "queries": queries}, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")

    def fetch_rising_queries(self, base_keywords: List[str], per_kw: int = 2) -> List[str]:
        """Fetch top related queries for base keywords using Google Trends API (pytrends)."""
        # Check cache first
        cached = self._load_cache()
        if cached:
            return cached

        dynamic: List[str] = []
        from pytrends.request import TrendReq
        import time

        try:
            pytrends = TrendReq(hl="en-US", tz=360)
            for kw in base_keywords:
                pytrends.build_payload([kw], timeframe="now 7-d", geo=self.geo)
                related = pytrends.related_queries()
                rq = related.get(kw, {})
                top_df = rq.get("top")
                if top_df is not None:
                    for _, row in top_df.head(per_kw).iterrows():
                        q = str(row["query"]).strip()
                        if q:
                            dynamic.append(q)
                # light throttle (2s per keyword)
                time.sleep(2)
        except Exception as e:
            print(f"⚠️ TrendAgent error: {e}")
            if not dynamic:
                print("⚠️ Using static keywords only (Trends blocked).")

        # dedupe
        seen = set()
        out: List[str] = []
        for q in dynamic:
            if q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)

        if out:
            self._save_cache(out)
        return out




class PlannerAgent:
    def __init__(self, static_keywords: List[str], trend_agent: TrendAgent, limit_total: int = 10):
        self.static_keywords = static_keywords
        self.trend_agent = trend_agent
        self.limit_total = limit_total

    def build_plan(self) -> Plan:
        print("🔍 Planner: fetching dynamic keywords from Trends...")
        dynamic = self.trend_agent.fetch_rising_queries(self.static_keywords, per_kw=2)
        merged = list(dict.fromkeys(self.static_keywords + dynamic))
        # cap total keywords used this run for cost control
        keywords = merged[: self.limit_total]
        limits = {
            "youtube": LIMIT_YOUTUBE_TOTAL,
            "reddit": LIMIT_REDDIT_TOTAL,
            "trends": LIMIT_TRENDS_TOTAL,
        }
        plan = Plan(keywords=keywords, limits=limits)
        ensure_dirs()
        with open(PLAN_PATH, "w") as f:
            json.dump({"keywords": plan.keywords, "limits": plan.limits, "generated_at": now_iso()}, f, indent=2)
        print(f"✅ Planner: plan saved → {PLAN_PATH}")
        return plan
    
import random

class AdaptivePlannerAgent:
    def __init__(self, performance_csv="output/performance_log.csv"):
        self.performance_csv = performance_csv

    def generate_plan(self, base_keywords, trend_keywords, caps={"youtube": 2, "reddit": 2}):
        """
        Returns a new plan: list of dicts like:
        {"platform": "youtube", "keyword": "elder care"}
        """
        print("🧠 AdaptivePlanner: building next run plan...")
        df_perf = None
        if os.path.exists(self.performance_csv):
            df_perf = pd.read_csv(self.performance_csv)
        else:
            print("⚠️ No previous performance data — using base plan.")

        weighted_keywords = []

        if df_perf is not None and not df_perf.empty:
            # Get success counts
            scores = (
                df_perf.groupby("Keyword")["SuccessCount"]
                .sum()
                .reset_index()
                .sort_values("SuccessCount", ascending=False)
            )

            # Weight keywords by success
            for _, row in scores.iterrows():
                weighted_keywords.extend([row["Keyword"]] * int(row["SuccessCount"]))

        # Always include fallback
        weighted_keywords = list(set(weighted_keywords + base_keywords + trend_keywords))
        random.shuffle(weighted_keywords)

        plan = []
        for platform, cap in caps.items():
            chosen = weighted_keywords[:cap]
            for kw in chosen:
                plan.append({"platform": platform, "keyword": kw})
                weighted_keywords.pop(0)

        print(f"✅ AdaptivePlanner: generated {len(plan)} tasks")
        return plan


class DiscoveryAgent:
    def __init__(self, yt_api_key: Optional[str], reddit_cfg: Dict[str, Optional[str]], store=None):
        self.yt_api_key = yt_api_key
        self.reddit_cfg = reddit_cfg
        self.store = store  # pass StoreAgent here for dedupe checking
        self.ytsvc = gbuild("youtube", "v3", developerKey=self.yt_api_key) if self.yt_api_key else None
        self.rcli = None
        if reddit_cfg.get("client_id") and reddit_cfg.get("client_secret") and reddit_cfg.get("user_agent"):
            self.rcli = praw.Reddit(
                client_id=reddit_cfg["client_id"],
                client_secret=reddit_cfg["client_secret"],
                user_agent=reddit_cfg["user_agent"],
            )
        self.reddit_subs = [
            "caregiving", "Alzheimers", "AgingParents", "nursing", "HomeHealthNursing", "digitalhealth"
        ]

    def discover_youtube(self, keywords: List[str], cap_total: int, conn=None) -> List[Dict[str, str]]:
        """Fetch up to cap_total unseen YouTube videos across keywords."""
        if not self.ytsvc:
            return []
        items, tried = [], set()
        import random

        for kw in keywords:
            if len(items) >= cap_total:
                break
            order_modes = ["relevance", "date", "viewCount"]
            order = random.choice(order_modes)
            next_page = None
            attempts = 0
            while len(items) < cap_total and attempts < 5:       # try up to 5 pages/rounds
                req = self.ytsvc.search().list(
                    q=kw,
                    part="snippet",
                    type="video",
                    maxResults=10,
                    order=order,
                    pageToken=next_page,
                )
                resp = req.execute()
                for it in resp.get("items", []):
                    vid = it["id"]["videoId"]
                    url = f"https://www.youtube.com/watch?v={vid}"
                    if url in tried:
                        continue
                    tried.add(url)
                    if conn and db_already_recent(conn, url, COOLDOWN_DAYS):
                        continue
                    title = it["snippet"]["title"]
                    items.append({"platform": "youtube", "keyword": kw, "topic": title, "url": url})
                    if len(items) >= cap_total:
                        break
                next_page = resp.get("nextPageToken")
                attempts += 1
                if not next_page:
                    break
        return items[:cap_total]


    def discover_reddit(self, keywords: List[str], cap_total: int, conn=None) -> List[Dict[str, str]]:
        """Fetch up to cap_total unseen Reddit posts across subreddits."""
        if not self.rcli:
            return []
        items, tried = [], set()
        import random
        sort_modes = ["top", "hot", "new"]

        for sub in self.reddit_subs:
            if len(items) >= cap_total:
                break
            subreddit = self.rcli.subreddit(sub)
            for kw in keywords:
                if len(items) >= cap_total:
                    break
                mode = random.choice(sort_modes)
                attempts = 0
                while len(items) < cap_total and attempts < 3:   # retry 3 search variants
                    results = subreddit.search(kw, sort=mode, time_filter="month", limit=10)
                    for post in results:
                        url = f"https://www.reddit.com{post.permalink}"
                        if url in tried:
                            continue
                        tried.add(url)
                        if conn and db_already_recent(conn, url, COOLDOWN_DAYS):
                            continue
                        items.append({
                            "platform": "reddit",
                            "keyword": kw,
                            "topic": post.title,
                            "url": url,
                        })
                        if len(items) >= cap_total:
                            break
                    attempts += 1
        return items[:cap_total]





class WriterAgent:
    def __init__(self, openai_key: Optional[str]):
        if openai_key and OpenAI is not None:
            self.client = OpenAI(api_key=openai_key)
        else:
            self.client = None

    @staticmethod
    def tone_for(platform: str) -> str:
        if platform == "reddit":
            return "casual, peer-to-peer, supportive, no marketing"
        if platform == "youtube":
            return "empathetic, concise, warm, encouraging, non-promotional"
        return "helpful, informative, respectful"

    def generate(self, platform: str, topic: str, brand_name: str = "CareYaya") -> str:
        if not self.client:
            # graceful fallback for demos if key not present
            return f"Thoughtful insight. At {brand_name}, we see families facing this every day and strive to help compassionately."
        tone = self.tone_for(platform)
        prompt = (
            f"You are a growth assistant for {brand_name}, a startup that helps families find trusted in-home caregivers. "
            f"Write a short (1–2 lines) {platform} comment for the topic '{topic}'. And invite them to check our services. "
            f"Tone: {tone}. Mention {brand_name} naturally without sounding promotional."
        )
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()


class StoreAgent:
    def __init__(self):
        self.conn = db_connect()
        db_init(self.conn)

    def skip_recent(self, url: str) -> bool:
        return db_already_recent(self.conn, url, COOLDOWN_DAYS)

    def persist(self, row: Dict[str, Any]):
        db_insert_post(self.conn, row)

    def append_csv(self, rows: List[Dict[str, Any]]):
        if not rows:
            return
        df_new = pd.DataFrame(rows)
        # normalize
        df_new["URL"] = df_new["URL"].astype(str).str.strip()
        df_new["Generated Comment"] = df_new["Generated Comment"].astype(str).str.replace("\n", " ").str.strip()
        df_new["Posted"] = df_new["Posted"].astype(bool)
        df_new["Posted At"] = df_new["Posted At"].fillna("")
        if os.path.exists(CSV_LOG):
            df_old = pd.read_csv(CSV_LOG)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all.drop_duplicates(subset=["URL", "Generated Comment"], inplace=True)
            df_all.to_csv(CSV_LOG, index=False)
        else:
            df_new.to_csv(CSV_LOG, index=False)
        # per-run log
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_new.to_csv(os.path.join(LOG_DIR, f"run_{run_stamp}.csv"), index=False)

import os, time, random, pickle, praw
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv
load_dotenv()


class PublisherAgent:
    """Posts comments on YouTube and Reddit with correct authentication and safe rate-limiting."""
    SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

    def __init__(self, reddit_cfg=None):
        self.yt_service = self._auth_youtube()       # interactive OAuth first time only
        self.rcli = self._auth_reddit(reddit_cfg)    # script-based login

    # ---------- YOUTUBE ----------
    def _auth_youtube(self):
        creds = None
        if os.path.exists("youtube_token.json"):
            with open("youtube_token.json", "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # force re-auth so we get the right scope
                flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", self.SCOPES)
                creds = flow.run_local_server(port=0, prompt="consent")
            with open("youtube_token.json", "wb") as token:
                pickle.dump(creds, token)
        service = build("youtube", "v3", credentials=creds)
        print("✅ YouTube OAuth complete and authorized for commenting.")
        return service

    def post_youtube_comment(self, video_url: str, text: str) -> bool:
        """Posts a comment to the given YouTube video."""
        try:
            video_id = video_url.split("v=")[-1]
            body = {
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {"snippet": {"textOriginal": text}},
                }
            }
            self.yt_service.commentThreads().insert(part="snippet", body=body).execute()
            print(f"💬 Posted YouTube comment: {video_url}")
            # light delay to stay under quota
            time.sleep(random.uniform(10, 20))
            return True
        except Exception as e:
            print(f"⚠️ YouTube posting error for {video_url}: {e}")
            return False

    # ---------- REDDIT ----------
    def _auth_reddit(self, reddit_cfg):
        """Logs into Reddit using a script app and username/password."""
        if not reddit_cfg or not reddit_cfg.get("client_id"):
            print("⚠️ Reddit credentials missing.")
            return None
        try:
            r = praw.Reddit(
                client_id=reddit_cfg["client_id"],
                client_secret=reddit_cfg["client_secret"],
                user_agent=reddit_cfg["user_agent"],
                username=reddit_cfg["username"],
                password=reddit_cfg["password"],
            )
            me = r.user.me()
            if me:
                print(f"🤖 Logged in to Reddit as: {me}")
                return r
            else:
                print("⚠️ Reddit login failed (user.me() returned None).")
                return None
        except Exception as e:
            print(f"⚠️ Reddit auth error: {e}")
            return None

    def post_reddit_comment(self, post_url: str, text: str) -> bool:
        """Posts a top-level comment on a Reddit submission."""
        if not self.rcli:
            print("⚠️ Reddit client not initialized.")
            return False
        try:
            submission = self.rcli.submission(url=post_url)
            submission.reply(text)
            print(f"💬 Posted Reddit comment: {post_url}")
            time.sleep(random.uniform(60, 120))  # 1-2 min delay between comments
            return True
        except Exception as e:
            print(f"⚠️ Reddit posting error for {post_url}: {e}")
            return False

class EvaluatorAgent:
    def __init__(self, posted_csv="output/comments_posted.csv", performance_csv="output/performance_log.csv"):
        self.posted_csv = posted_csv
        self.performance_csv = performance_csv

    def evaluate(self):
        if not os.path.exists(self.posted_csv):
            print("⚠️ No posted comments file found.")
            return

        df = pd.read_csv(self.posted_csv)
        if df.empty:
            print("⚠️ No data to evaluate.")
            return

        # Standardize column names
        df.columns = [c.strip() for c in df.columns]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Posted"] = df["Posted"].astype(str)

        # Basic success metric — simple count of posted=True per keyword/platform
        perf = (
            df[df["Posted"].str.lower() == "true"]
            .groupby(["Platform", "Keyword"])
            .size()
            .reset_index(name="SuccessCount")
        )

        # Add run metadata
        perf["EvaluatedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append to performance log
        if os.path.exists(self.performance_csv):
            old = pd.read_csv(self.performance_csv)
            perf = pd.concat([old, perf], ignore_index=True)

        perf.to_csv(self.performance_csv, index=False)
        print(f"✅ Evaluation complete → {self.performance_csv}")

# ----------------------
# Orchestrator
# ----------------------
class AgenticEngine:
    def __init__(self, cfg: Dict[str, Optional[str]]):
        self.cfg = cfg
        self.trend_agent = TrendAgent()
        self.planner = PlannerAgent(DEFAULT_KEYWORDS, self.trend_agent, limit_total=10)
        self.discovery = DiscoveryAgent(
            yt_api_key=cfg.get("YOUTUBE_API_KEY"),
            reddit_cfg={
                "client_id": cfg.get("REDDIT_CLIENT_ID"),
                "client_secret": cfg.get("REDDIT_CLIENT_SECRET"),
                "user_agent": cfg.get("REDDIT_USER_AGENT"),
            },
        )
        self.writer = WriterAgent(cfg.get("OPENAI_API_KEY"))
        self.store = StoreAgent()
        self.publisher = PublisherAgent(
            reddit_cfg={
                "client_id": os.getenv("REDDIT_CLIENT_ID"),
                "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
                "user_agent": os.getenv("REDDIT_USER_AGENT"),
                "username": os.getenv("REDDIT_USERNAME"),
                "password": os.getenv("REDDIT_PASSWORD"),
            }
        )



    def plan(self) -> Plan:
        return self.planner.build_plan()

    def run_once(self, plan):
        """
        Supports both the old Plan object and new adaptive plan list.
        """
        # --- Compatibility layer ---
        if isinstance(plan, list):
            # Adaptive plan: list of dicts
            tasks = plan
            # infer caps dynamically
            caps = {
                "youtube": sum(1 for t in tasks if t["platform"] == "youtube"),
                "reddit": sum(1 for t in tasks if t["platform"] == "reddit"),
            }
            keywords = list({t["keyword"] for t in tasks})
        else:
            # Legacy Plan object
            caps = plan.limits
            keywords = plan.keywords
            tasks = None

        print("🧭 Engine: starting run with plan caps:", caps)

        # --- Discovery Phase ---
        discovered: List[Dict[str, str]] = []
        if tasks is None:
            yt_items = self.discovery.discover_youtube(keywords, caps["youtube"], conn=self.store.conn)
            rd_items = self.discovery.discover_reddit(keywords, caps["reddit"], conn=self.store.conn)
        else:
            # Adaptive tasks already specify platforms + keywords
            yt_items, rd_items = [], []
            for task in tasks:
                kw, platform = task["keyword"], task["platform"]
                if platform == "youtube":
                    yt_items.extend(self.discovery.discover_youtube([kw], 1, conn=self.store.conn))
                elif platform == "reddit":
                    rd_items.extend(self.discovery.discover_reddit([kw], 1, conn=self.store.conn))

        # skip pseudo-trend entries entirely
        discovered.extend(yt_items)
        discovered.extend(rd_items)

        # --- Deduplication Phase ---
        seen = set()
        queue: List[Dict[str, str]] = []
        for it in discovered:
            url = it["url"]
            if url in seen:
                continue
            seen.add(url)
            if self.store.skip_recent(url):
                print(f"⏩ Skip (cooldown): {it['platform']} | {it['topic'][:60]}")
                continue
            queue.append(it)

        print(f"📦 Queue size: {len(queue)} items")

        # --- Generation + Publishing Phase ---
        rows_for_csv: List[Dict[str, Any]] = []
        processed = 0
        for it in queue:
            platform = it["platform"]
            if platform not in ["youtube", "reddit"]:
                continue

            topic = it["topic"]
            keyword = it.get("keyword")
            url = it["url"]

            print(f"✍️  Writing for {platform}: {topic[:60]} ...")
            try:
                text = self.writer.generate(platform=platform, topic=topic)
            except Exception as e:
                print(f"⚠️ Writer error: {e}")
                continue

            is_posted = False
            posted_time = None

            if platform == "youtube":
                is_posted = self.publisher.post_youtube_comment(url, text)
            elif platform == "reddit":
                is_posted = self.publisher.post_reddit_comment(url, text)

            if is_posted:
                posted_time = now_iso()

            self.store.persist({
                "platform": platform,
                "keyword": keyword,
                "topic": topic,
                "url": url,
                "generated_text": text,
                "posted": is_posted,
                "posted_at": posted_time,
            })

            rows_for_csv.append({
                "Timestamp": now_iso(),
                "Platform": platform,
                "Keyword": str(keyword),
                "Topic": str(topic),
                "URL": str(url),
                "Generated Comment": str(text),
                "Posted": is_posted,
                "Posted At": posted_time if is_posted else "",
            })
            processed += 1

        self.store.append_csv(rows_for_csv)
        print(f"\n✅ Run complete — processed {processed} items. CSV: {CSV_LOG}")



# ----------------------
# Entrypoint / CLI
# ----------------------

def load_config() -> Dict[str, Optional[str]]:
    load_dotenv()
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "YOUTUBE_API_KEY": os.getenv("YOUTUBE_API_KEY"),
        "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID"),
        "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET"),
        "REDDIT_USER_AGENT": os.getenv("REDDIT_USER_AGENT"),
    }


def main():
    ensure_dirs()
    cfg = load_config()
    engine = AgenticEngine(cfg)

    args = sys.argv[1:]
    loop_mode = "--loop" in args  # optional flag for continuous operation

    while True:
        print("\n🚀 Starting new Agentic Growth Cycle...")

        # 1️⃣ Step 1: Evaluate previous run
        evaluator = EvaluatorAgent()
        evaluator.evaluate()

        # 2️⃣ Step 2: Create adaptive plan
        adaptive_planner = AdaptivePlannerAgent()
        base_keywords = ["elder care", "caregiver burnout", "AI caregiving"]
        trend_keywords = []
        try:
            trend_keywords = engine.trend_agent.fetch_rising_queries(base_keywords) or []
        except Exception as e:
            print(f"⚠️ Trend fetch failed: {e}")

        plan_data = adaptive_planner.generate_plan(
            base_keywords=base_keywords,
            trend_keywords=trend_keywords,
            caps={"youtube": 2, "reddit": 2}
        )

        # 3️⃣ Step 3: Execute the run
        engine.run_once(plan_data)

        # 4️⃣ Step 4: Decide whether to loop or stop
        if loop_mode:
            print("🕒 Sleeping 4 hours before next adaptive cycle...\n")
            time.sleep(4 * 60 * 60)
        else:
            break

if __name__ == "__main__":
    main()