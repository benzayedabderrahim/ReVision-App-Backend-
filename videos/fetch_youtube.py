import os
import sys
import django
import requests
import logging
import time
from datetime import datetime
from dateutil import parser
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "youtube_app.settings")
django.setup()

from videos.models import Video, Comment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("youtube_fetcher.log"),
        logging.StreamHandler()
    ]
)

# YouTube API configuration
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    logging.error("""
    YouTube API key not found in environment variables.
    Please create a .env file in your project directory with:
    YOUTUBE_API_KEY=your_actual_key_here
    """)
    sys.exit(1)

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEO_DETAILS_URL = "https://www.googleapis.com/youtube/v3/videos"
COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

# Initialize sentiment analyzer
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    logging.error(f"Failed to initialize sentiment analyzer: {e}")
    sys.exit(1)

def analyze_sentiment(text):
    """Analyze text sentiment."""
    try:
        result = sentiment_analyzer(text[:512])  # Truncate to model limit
        return result[0]["label"] == "POSITIVE", result[0]["score"]
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        return False, 0.0

def fetch_search_results(query, page_token=None):
    """Fetch search results from YouTube API."""
    try:
        params = {
            "part": "snippet",
            "maxResults": 50,
            "q": query,
            "type": "video",
            "order": "relevance",
            "key": API_KEY
        }
        if page_token:
            params["pageToken"] = page_token
            
        response = requests.get(SEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching search results: {e}")
        return None

def fetch_video_details(video_id):
    """Fetch video details including statistics."""
    try:
        response = requests.get(
            VIDEO_DETAILS_URL,
            params={
                "part": "snippet,statistics",
                "id": video_id,
                "key": API_KEY
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])[0] if data.get("items") else None
    except Exception as e:
        logging.error(f"Error fetching video details for {video_id}: {e}")
        return None

def fetch_video_comments(video_id, max_results=100):
    """Fetch comments for a video."""
    comments = []
    next_page_token = None
    
    try:
        while len(comments) < max_results:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, max_results - len(comments)),
                "textFormat": "plainText",
                "key": API_KEY
            }
            if next_page_token:
                params["pageToken"] = next_page_token
                
            response = requests.get(COMMENTS_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("items", []):
                try:
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append(comment["textDisplay"])
                except KeyError:
                    continue
                    
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break
                
            time.sleep(1)  # Respect YouTube API rate limits
                
    except Exception as e:
        logging.error(f"Error fetching comments for {video_id}: {e}")
        
    return comments[:max_results]

def process_video(video_id, query):
    """Process and save a single video with its comments."""
    try:
        existing = Video.objects.filter(video_id=video_id).first()
        if existing and (datetime.now() - existing.updated_at).days < 7:
            return
            
        video_data = fetch_video_details(video_id)
        if not video_data:
            return
            
        snippet = video_data.get("snippet", {})
        stats = video_data.get("statistics", {})
        
        comments = fetch_video_comments(video_id)
        positive_comments = []
        
        for comment_text in comments:
            is_positive, confidence = analyze_sentiment(comment_text)
            if is_positive and confidence > 0.85:
                positive_comments.append(comment_text)
        
        if not positive_comments:
            return
            
        video_defaults = {
            "title": snippet.get("title", "No Title"),
            "description": snippet.get("description", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments_count": len(comments),
            "positive_comments_count": len(positive_comments),
            "upload_date": parser.parse(snippet.get("publishedAt", datetime.now().isoformat())),
            "query": query,
        }
        
        video, created = Video.objects.update_or_create(
            video_id=video_id,
            defaults=video_defaults
        )
        
        Comment.objects.filter(video=video).delete()
        for comment_text in positive_comments:
            Comment.objects.create(
                video=video,
                text=comment_text,
                is_positive=True
            )
            
        logging.info(f"{'Created' if created else 'Updated'} video: {video.title}")
        
    except Exception as e:
        logging.error(f"Error processing video {video_id}: {e}")

def fetch_and_save_videos(query, max_videos=20):
    """Main function to fetch and save videos."""
    page_token = None
    videos_processed = 0
    
    while videos_processed < max_videos:
        search_data = fetch_search_results(query, page_token)
        if not search_data or not search_data.get("items"):
            break
            
        video_ids = [
            item["id"]["videoId"] 
            for item in search_data.get("items", []) 
            if item.get("id", {}).get("videoId")
        ]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            for video_id in video_ids:
                executor.submit(process_video, video_id, query)
                time.sleep(1)
                
        videos_processed += len(video_ids)
        page_token = search_data.get("nextPageToken")
        if not page_token or videos_processed >= max_videos:
            break

if __name__ == "__main__":
    try:
        queries = ["Algorithm tunisie"]
        for query in queries:
            logging.info(f"Processing query: {query}")
            fetch_and_save_videos(query)
            time.sleep(5)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)