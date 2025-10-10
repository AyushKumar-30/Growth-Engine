import os, praw
from dotenv import load_dotenv

load_dotenv()  # reads .env in this folder

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
)

print("Logged in as:", reddit.user.me())
url = "https://www.reddit.com/r/dotnet/comments/1nedu7c/what_really_differentiates_junior_mid_and_senior/"
submission = reddit.submission(url=url)
submission.reply("Test comment from CareYaya Engine 🤖")
print("✅ Comment posted")
