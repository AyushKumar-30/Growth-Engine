"""
CareYaya Growth Engine — Prototype (single-file)

What this does
--------------
1) Discovers high-signal caregiving topics across YouTube, Reddit, and Google Trends
2) Generates context-aware, platform-appropriate comments/posts with OpenAI
3) Deduplicates & enforces cooldowns via SQLite
4) Optionally posts to YouTube (OAuth), logs all activity, and appends a CSV for your dashboard
5) Designed to run headless every 4 hours (cron / GitHub Actions)

Quick start
-----------
1) pip install -r requirements.txt  (see requirements block below)
2) Put credentials into .env (see ENV section below)
3) (Optional, for posting) download Google OAuth client as client_secret.json
4) python careyaya_engine.py  (dry-run by default)

Schedule every 4 hours (cron example)
-------------------------------------
0 */4 * * * /usr/bin/python3 /path/to/careyaya_engine.py >> output/logs/cron.log 2>&1

Requirements
------------
openai
python-dotenv
pandas
requests
google-api-python-client
google-auth-oauthlib
google-auth-httplib2
praw
pytrends

ENV (.env)
----------
OPENAI_API_KEY=sk-...
YOUTUBE_API_KEY=AIza...
# For YouTube OAuth posting
# Place client_secret.json in project root

# Reddit (script app) — create at https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=careyaya-growth-engine/0.1 by your_reddit_username
REDDIT_USERNAME=your_reddit_username   # (only needed if posting to Reddit)
REDDIT_PASSWORD=your_reddit_password   # (only needed if posting to Reddit)

CAREYAYA_LINK=https://careyaya.com

Notes
-----
• Links are NOT embedded in comments to avoid filters; brand is referenced by name.
• Safe de-linked mention format is available if you want (see build_safe_brand_mention).
"""

import os
import sys
import csv
import time
import json
import math
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# YouTube Search API (for discovery)
from googleapiclient.discovery import build as gbuild

# YouTube OAuth posting
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build as ybuild
import pickle

# Reddit
import praw

# Google Trends
from pytrends.request import TrendReq

# ---------------------------
# Configuration (edit freely)
# ---------------------------
DEFAULT_KEYWORDS = [
    "elder care",
    "caregiver burnout",
    "home health care",
    "dementia caregiving",
    "AI caregiving",
]

PLATFORMS_ENABLED = {
    "youtube": True,   # discovery always on; posting controlled by POST_TO_YOUTUBE
    "reddit": True,    # discovery on; posting off by default
    "trends": True,
}

POST_TO_YOUTUBE = False  # set True to actually post comments via OAuth
POST_TO_REDDIT = False    # prototype keeps this False; discovery only

MAX_YT_RESULTS = 2   # per run
MAX_REDDIT_RESULTS = 2
COOLDOWN_DAYS = 7     # do not re-comment the same URL within this window

SQLITE_PATH = os.path.join("output", "engine.db")
CSV_LOG = os.path.join("output", "comments_posted.csv")
LOG_DIR = os.path.join("output", "logs")
TOKEN_PATH = "token.pickle"  # YouTube OAuth refresh token
CLIENT_SECRET_JSON = "client_secret.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

# -----------------------------------
# Dynamic Keyword Expansion (Trends)
# -----------------------------------

def get_dynamic_keywords(static_keywords, limit=10):
    """
    Expands the static caregiving keywords with rising queries from Google Trends.
    Returns a deduplicated list of up to `limit` total keywords.
    """
    dynamic = []
    try:
        print("🔍 Fetching dynamic keywords from Google Trends...")
        trends = discover_trends(static_keywords)
        # Extract rising query text
        dynamic = [
            item["topic"].replace("Rising query: ", "").strip()
            for item in trends
            if item.get("topic")
        ]
        print(f"✅ Found {len(dynamic)} trending topics.")
    except Exception as e:
        print(f"⚠️ Trends expansion failed: {e}")

    # Merge and deduplicate
    all_keywords = list(dict.fromkeys(static_keywords + dynamic))
    # Limit to avoid too many API calls downstream
    final_kw = all_keywords[:limit]
    print(f"🧠 Final keyword set for this run: {final_kw}")
    return final_kw

# --------------
# Util functions
# --------------

def ensure_dirs():
    os.makedirs("output", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -----------------------
# Database (SQLite) layer
# -----------------------
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
        "SELECT 1 FROM posts WHERE url=? AND posted=1 AND posted_at >= ? LIMIT 1",
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


# ------------------
# Auth / Credentials
# ------------------

def load_env():
    load_dotenv()
    cfg = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "YOUTUBE_API_KEY": os.getenv("YOUTUBE_API_KEY"),
        "CAREYAYA_LINK": os.getenv("CAREYAYA_LINK", "https://careyaya.com"),
        "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID"),
        "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET"),
        "REDDIT_USER_AGENT": os.getenv("REDDIT_USER_AGENT", "careyaya-growth-engine/0.1"),
        "REDDIT_USERNAME": os.getenv("REDDIT_USERNAME"),
        "REDDIT_PASSWORD": os.getenv("REDDIT_PASSWORD"),
    }
    return cfg


def youtube_search_service(api_key: str):
    return gbuild("youtube", "v3", developerKey=api_key)


def youtube_oauth_service() -> Any:
    """Authenticate YouTube for posting; persists token to avoid repeated logins."""
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_JSON, SCOPES)
            creds = flow.run_local_server(port=8080, prompt="consent")
        with open(TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)
    return ybuild("youtube", "v3", credentials=creds)


def reddit_client(cfg) -> Optional[praw.Reddit]:
    if not (cfg["REDDIT_CLIENT_ID"] and cfg["REDDIT_CLIENT_SECRET"] and cfg["REDDIT_USER_AGENT"]):
        return None
    return praw.Reddit(
        client_id=cfg["REDDIT_CLIENT_ID"],
        client_secret=cfg["REDDIT_CLIENT_SECRET"],
        user_agent=cfg["REDDIT_USER_AGENT"],
        username=cfg.get("REDDIT_USERNAME"),
        password=cfg.get("REDDIT_PASSWORD"),
    )


# ----------------------
# Discovery: YouTube API
# ----------------------

def discover_youtube(ytsvc, keywords: List[str], max_results: int = 2) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for kw in keywords:
        req = ytsvc.search().list(
            q=kw,
            part="snippet",
            type="video",
            maxResults=min(max_results, 50),
            order="viewCount",
        )
        resp = req.execute()
        for it in resp.get("items", []):
            vid = it["id"]["videoId"]
            title = it["snippet"]["title"]
            url = f"https://www.youtube.com/watch?v={vid}"
            items.append({"platform": "youtube", "keyword": kw, "topic": title, "url": url})
    return items


# --------------------
# Discovery: Reddit API
# --------------------
REDDIT_SUBS = [
    "caregiving",
    "Alzheimers",
    "AgingParents",
    "nursing",
    "HomeHealthNursing",
    "digitalhealth",
]


def discover_reddit(r: praw.Reddit, keywords: List[str], limit: int = 1) -> List[Dict[str, str]]:
    if r is None:
        return []
    results: List[Dict[str, str]] = []
    for sub in REDDIT_SUBS:
        try:
            subreddit = r.subreddit(sub)
            for kw in keywords:
                for post in subreddit.search(kw, sort="top", time_filter="week", limit=limit):
                    results.append({
                        "platform": "reddit",
                        "keyword": kw,
                        "topic": post.title,
                        "url": f"https://www.reddit.com{post.permalink}",
                    })
        except Exception:
            continue
    return results


# -----------------------
# Discovery: Google Trends
# -----------------------

def discover_trends(keywords: List[str]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        for kw in keywords:
            pytrends.build_payload([kw], timeframe='now 7-d', geo='US')
            related = pytrends.related_queries()
            rq = related.get(kw, {})
            top_df = rq.get('top')
            if top_df is not None:
                # take top 5
                for _, row in top_df.head(2).iterrows():
                    topic = row['query']
                    items.append({
                        "platform": "trends",
                        "keyword": kw,
                        "topic": f"Rising query: {topic}",
                        "url": f"https://trends.google.com/trends/explore?q={requests.utils.quote(topic)}",
                    })
    except Exception:
        pass
    return items


# ------------------------------
# Generation: OpenAI (tone-aware)
# ------------------------------

def openai_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai package not installed. pip install openai")
    return OpenAI(api_key=api_key)


def platform_tone(platform: str) -> str:
    if platform == "reddit":
        return "casual, peer-to-peer, supportive, no marketing"
    if platform == "youtube":
        return "empathetic, concise, warm, encouraging, non-promotional"
    return "helpful, informative, respectful"


def build_safe_brand_mention(brand_link: str) -> str:
    # return a de-linked safe mention to avoid filters
    safe = brand_link.replace("https://", "").replace("http://", "").replace(".", " dot ")
    return f"(learn more at {safe})"


def generate_text(client: OpenAI, platform: str, topic: str, brand_name: str = "CareYaya", include_safe_mention: bool = False, brand_link: str = "https://careyaya.com") -> str:
    tone = platform_tone(platform)
    sys_prompt = (
        f"You are a growth assistant who talks like a real human for {brand_name}, a startup that helps families find trusted in-home caregivers. "
        f"Write a short comment (1–2 lines) for {platform} on the topic '{topic}'. "
        f"Tone: {tone}. Add a call to action and mention website url. Mention {brand_name} naturally without sounding promotional."
    )
    msg = [{"role": "user", "content": sys_prompt}]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msg,
        max_tokens=80,
        temperature=0.8,
    )
    text = resp.choices[0].message.content.strip()
    if include_safe_mention:
        text += "\n" + build_safe_brand_mention(brand_link)
    return text


# -----------------
# Posting: YouTube
# -----------------

def yt_video_id_from_url(url: str) -> Optional[str]:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    if "/shorts/" in url:
        return url.rsplit("/", 1)[-1]
    return None


def post_comment_youtube(ytsvc_oauth, video_id: str, text: str) -> Dict[str, Any]:
    req = ytsvc_oauth.commentThreads().insert(
        part="snippet",
        body={
            "snippet": {
                "videoId": video_id,
                "topLevelComment": {"snippet": {"textOriginal": text}},
            }
        },
    )
    return req.execute()


# -----------------------------
# Engine: orchestrate one cycle
# -----------------------------

def run_engine():
    ensure_dirs()
    cfg = load_env()
    conn = db_connect()
    db_init(conn)

    # Expand keyword list dynamically
    keywords = get_dynamic_keywords(DEFAULT_KEYWORDS)

    # Services
    yts = youtube_search_service(cfg["YOUTUBE_API_KEY"]) if PLATFORMS_ENABLED["youtube"] else None
    yt_oauth = youtube_oauth_service() if (POST_TO_YOUTUBE and PLATFORMS_ENABLED["youtube"]) else None
    rcli = reddit_client(cfg) if PLATFORMS_ENABLED["reddit"] else None
    oai = openai_client(cfg["OPENAI_API_KEY"]) if cfg["OPENAI_API_KEY"] else None

    # 1) Discover
    discovered: List[Dict[str, str]] = []
    if yts:
        discovered += discover_youtube(yts, keywords, MAX_YT_RESULTS)
    if rcli:
        discovered += discover_reddit(rcli, keywords, MAX_REDDIT_RESULTS)
    if PLATFORMS_ENABLED["trends"]:
        discovered += discover_trends(keywords)

    # Deduplicate by URL
    seen_urls = set()
    uniq: List[Dict[str, str]] = []
    for it in discovered:
        if it["url"] not in seen_urls:
            seen_urls.add(it["url"])
            uniq.append(it)

    rows_for_csv: List[Dict[str, Any]] = []
    processed = 0

    for item in uniq:
        url = item["url"]
        platform = item["platform"]
        keyword = item.get("keyword")
        topic = item.get("topic")

        # Cooldown skip
        if db_already_recent(conn, url, COOLDOWN_DAYS):
            print(f"⏩ Skip (cooldown): {platform} | {topic}")
            continue

        # 2) Generate text
        try:
            text = generate_text(
                client=oai,
                platform=platform,
                topic=topic,
                include_safe_mention=False,
                brand_link=cfg["CAREYAYA_LINK"],
            )
        except Exception as e:
            print(f"⚠️ OpenAI generation failed: {e}")
            continue

        posted = False
        posted_at = None

        # 3) Post (YouTube only in prototype)
        if platform == "youtube" and POST_TO_YOUTUBE and yt_oauth is not None:
            vid = yt_video_id_from_url(url)
            if vid:
                try:
                    post_comment_youtube(yt_oauth, vid, text)
                    posted = True
                    posted_at = now_iso()
                    print(f"✅ Posted YouTube comment: {topic}")
                except Exception as e:
                    print(f"⚠️ YouTube post failed: {e}")

        # 4) Persist (even if not posted yet — keeps queue/history)
        db_insert_post(conn, {
            "platform": platform,
            "keyword": keyword,
            "topic": topic,
            "url": url,
            "generated_text": text,
            "posted": posted,
            "posted_at": posted_at,
        })

        rows_for_csv.append({
            "Timestamp": now_iso(),
            "Platform": str(platform),
            "Keyword": str(keyword),
            "Topic": str(topic),
            "URL": str(url).strip(),
            "Generated Comment": str(text).replace("\n", " ").strip(),
            "Posted": bool(posted),
            "Posted At": str(posted_at) if posted_at else "",
        })
        processed += 1

    # 5) Append CSV log for dashboard
    if rows_for_csv:
        df_new = pd.DataFrame(rows_for_csv)
        if os.path.exists(CSV_LOG):
            df_old = pd.read_csv(CSV_LOG)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            # drop exact duplicates
            df_all.drop_duplicates(subset=["URL", "Generated Comment"], inplace=True)
            df_all.to_csv(CSV_LOG, index=False)
        else:
            df_new.to_csv(CSV_LOG, index=False)

        # per-run log
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_new.to_csv(os.path.join(LOG_DIR, f"run_{run_stamp}.csv"), index=False)

    print(f"\n📊 Run complete — processed {processed} items. Log: {CSV_LOG}")


if __name__ == "__main__":
    try:
        run_engine()
    except KeyboardInterrupt:
        print("Interrupted.")
