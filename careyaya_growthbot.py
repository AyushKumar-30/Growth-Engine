import os
import requests
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
from openai import OpenAI
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pickle


SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]


# Load environment variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BITLY_TOKEN = os.getenv("BITLY_TOKEN")
CAREYAYA_LINK = os.getenv("CAREYAYA_LINK", "https://careyaya.com")

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize YouTube API
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def get_trending_videos(keyword, max_results=2):
    """Fetch top videos from YouTube for a given keyword"""
    request = youtube.search().list(
        q=keyword,
        part="snippet",
        type="video",
        maxResults=max_results,
        order="relevance"
    )
    response = request.execute()

    videos = []
    for item in response["items"]:
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append({"title": title, "url": url})
    return videos


def shorten_link(original_link):
    """Shorten URL using Bitly API"""
    if not BITLY_TOKEN:
        return original_link
    headers = {"Authorization": f"Bearer {BITLY_TOKEN}"}
    data = {"long_url": original_link}
    resp = requests.post("https://api-ssl.bitly.com/v4/shorten", json=data, headers=headers)
    if resp.status_code == 200:
        return resp.json()["link"]
    return original_link


def generate_comment(video_title):
    """Generate YouTube comment using OpenAI"""
    prompt = f"""
    You are a growth assistant for CareYaya, a healthcare startup helping families find trusted caregivers.
    Write a short, empathetic YouTube comment (1–2 lines) responding to the video titled '{video_title}'.
    Make it sound genuine, conversational, and natural — invite them to visit our website without including a link, but you can mention CareYaya by name.
    """


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

def get_authenticated_service():
    creds = None

    # Check if we already have a saved token
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    # If token is invalid or doesn’t exist, request new one
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=8080)
        # Save token for next time
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build("youtube", "v3", credentials=creds)

def post_comment(youtube_service, video_id, comment_text):
    """Post comment under a specific YouTube video"""
    request = youtube_service.commentThreads().insert(
        part="snippet",
        body={
            "snippet": {
                "videoId": video_id,
                "topLevelComment": {"snippet": {"textOriginal": comment_text}}
            }
        }
    )
    response = request.execute()
    return response

def main():
    print("🚀 Starting CareYaya GrowthBot...")

    # Authenticate with YouTube once (opens browser the first time)
    youtube_service = get_authenticated_service()

    keyword = input("Enter keyword to search YouTube (e.g. caregiving, elder care): ")
    videos = get_trending_videos(keyword)

    print(f"🔍 Found {len(videos)} videos. Generating and posting comments...")
    results = []

    for video in videos:
        title = video["title"]
        url = video["url"]
        video_id = url.split("v=")[-1]

        print(f"💬 Generating comment for: {title}")
        comment = generate_comment(title)

        print(f"📤 Posting comment to YouTube video: {video_id}")
        try:
            post_comment(youtube_service, video_id, comment)
            print("✅ Comment posted successfully!")
        except Exception as e:
            print(f"⚠️ Failed to post comment: {e}")

        # Optional shortened link for tracking
        link = shorten_link(CAREYAYA_LINK)

        results.append({
            "Video Title": title,
            "Video URL": url,
            "Generated Comment": comment,
            "CareYaya Link": link
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    output_path = "output/comments_posted.csv"
    if os.path.exists(output_path):
        old_df = pd.read_csv(output_path)
        combined_df = pd.concat([old_df, df], ignore_index=True)
        combined_df.drop_duplicates(subset=["Video URL", "Generated Comment"], inplace=True)
        combined_df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"\n📊 All done! Comments saved to {output_path}")
    print(df.head())
    print(df.head())


if __name__ == "__main__":
    main()
