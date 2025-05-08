import json
import time
from googleapiclient.discovery import build
from venv import logger
import os
from firebase_admin import exceptions as firebase_exceptions
from firebase_admin import auth as firebase_auth
from firebase_admin import db as firebase_db
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User
from django.shortcuts import render, get_object_or_404
from .models import RegistrationCode, Video, Comment
from transformers import pipeline
from django.core.paginator import Paginator
from django.db.models import Q
from django.db.models import F
import difflib
import base64
import json
import matplotlib.pyplot as plt
import io
from textblob import TextBlob
import re
from bs4 import BeautifulSoup
import requests
import logging
from django.conf import settings
from collections import Counter
import math
from .models import Video, VideoSimilarity
from rest_framework.decorators import api_view
from rest_framework.response import Response
from difflib import SequenceMatcher

# Initialize sentiment analysis pipeline
try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except Exception as e:
    print(f"Failed to initialize sentiment analyzer: {e}")
    sentiment_analyzer = None

# ===== Original Authentication Views =====
@csrf_exempt
def firebase_login(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            email = data.get("email")
            password = data.get("password")

            if not email or not password:
                return JsonResponse({"error": "Email and password are required"}, status=400)

            try:
                user = firebase_auth.get_user_by_email(email)
                
                user_ref = firebase_db.reference(f'users/{user.uid}')
                
                user_data = {
                    'email': user.email,
                    'last_login': firebase_db.ServerValue.TIMESTAMP,
                    'account_status': 'active'
                }
                
                user_ref.update(user_data)
                
                return JsonResponse({
                    "status": "success",
                    "user_id": user.uid,
                    "email": user.email
                })
                
            except firebase_exceptions.NotFoundError:
                return JsonResponse({"error": "Invalid credentials"}, status=401)
            except Exception as e:
                return JsonResponse({"error": str(e)}, status=400)
                
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Only POST method is allowed"}, status=405)

@csrf_exempt
def verify_code(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            email = data.get("email")
            code = data.get("code")

            if not email or not code:
                return JsonResponse({"error": "Email and code are required"}, status=400)

            user = User.objects.get(email=email)
            registration_code = RegistrationCode.objects.get(user=user)

            if registration_code.code == code:
                registration_code.delete()
                return JsonResponse({"message": "Code verified successfully!"})
            return JsonResponse({"error": "Invalid code"}, status=400)
        except User.DoesNotExist:
            return JsonResponse({"error": "User not found"}, status=404)
        except RegistrationCode.DoesNotExist:
            return JsonResponse({"error": "No registration code found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Invalid request method"}, status=405)

# signup 
@csrf_exempt
def signup(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            password = data.get('password')
            username = data.get('username')  # Get username from request

            if not email or not password or not username:
                return JsonResponse({'error': 'All fields are required'}, status=400)

            try:
                validate_email(email)
            except ValidationError:
                return JsonResponse({'error': 'Invalid email format'}, status=400)

            if User.objects.filter(email=email).exists():
                return JsonResponse({'error': 'Email already exists'}, status=400)

            if User.objects.filter(username=username).exists():
                return JsonResponse({'error': 'Username already exists'}, status=400)

            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            return JsonResponse({'message': 'Signup successful!'}, status=201)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

# ===== Original Video API Views =====
@csrf_exempt
def videos(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            return JsonResponse({"message": "Video processed successfully!"})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Method not allowed"}, status=405)

import os
import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Recherche (related videos)

@csrf_exempt
def search(request):
    if request.method == "POST":
        try:
            # Load request data
            data = json.loads(request.body)
            search_query = data.get("prompt", "").strip()
            quick_mode = data.get("quick_mode", True)

            if not search_query:
                return JsonResponse({"error": "Search query cannot be empty"}, status=400)

            # Get YouTube API key from environment
            YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
            if not YOUTUBE_API_KEY:
                logger.error("YouTube API key not configured")
                return JsonResponse({"error": "Service configuration error"}, status=500)

            # Initialize YouTube API client
            youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, cache_discovery=False)

            # Check if we already have matching videos in DB
            cached_videos = Video.objects.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query)
            ).order_by('-views')[:20]

            if cached_videos.exists() and len(cached_videos) >= 5:
                # Use cached video data
                video_data = [{
                    "video_id": video.video_id,
                    "title": video.title,
                    "description": video.description,
                    "url": video.url,
                    "thumbnail": str(video.thumbnail),
                    "views": video.views,
                    "likes": video.likes,
                    "comments_count": video.comments_count,
                    "published_at": video.upload_date.strftime("%Y-%m-%dT%H:%M:%SZ") if video.upload_date else "",
                    "embedded_link": f"https://www.youtube.com/embed/{video.video_id}",
                    "positive_comments_count": video.positive_comments_count,
                    # Removed 'negative_comments_count'
                    "neutral_comments_count": video.comments_count - video.positive_comments_count,
                    "positivity_percentage": (video.positive_comments_count / video.comments_count * 100) if video.comments_count > 0 else 0
                } for video in cached_videos]

                return JsonResponse({
                    "status": "success",
                    "count": len(video_data),
                    "videos": video_data,
                    "source": "cache"
                })

            # If not enough cached results, use YouTube API
            search_response = youtube.search().list(
                q=search_query,
                part='id,snippet',
                maxResults=20,
                type='video',
                order='viewCount'
            ).execute()

            # Extract video IDs from search results
            video_ids = []
            for item in search_response.get('items', []):
                if 'videoId' in item.get('id', {}):
                    video_ids.append(item['id']['videoId'])
                if len(video_ids) >= 20:
                    break

            if not video_ids:
                return JsonResponse({"videos": [], "message": "No videos found"})

            # Get detailed video info
            videos_response = youtube.videos().list(
                part='snippet,statistics',
                id=','.join(video_ids),
                maxResults=20
            ).execute()

            video_data = []

            for item in videos_response.get('items', []):
                try:
                    video_id = item['id']
                    stats = item['statistics']

                    # Check if we already have this video stored
                    existing_video = Video.objects.filter(video_id=video_id).first()

                    if existing_video:
                        # Use stored analysis
                        analysis = {
                            "positive_comments_count": existing_video.positive_comments_count,
                            # Removed 'negative_comments_count'
                            "neutral_comments_count": existing_video.comments_count - existing_video.positive_comments_count,
                            "positivity_percentage": (existing_video.positive_comments_count / existing_video.comments_count * 100)
                            if existing_video.comments_count > 0 else 0
                        }
                    else:
                        # Default analysis if not yet analyzed
                        analysis = {
                            "positive_comments_count": 0,
                            "neutral_comments_count": 0,
                            "positivity_percentage": 0
                        }

                    video_data.append({
                        "video_id": video_id,
                        "title": item['snippet'].get('title', 'Untitled'),
                        "description": item['snippet'].get('description', ''),
                        "url": f"https://youtube.com/watch?v={video_id}",
                        "thumbnail": item['snippet']['thumbnails']['high']['url'],
                        "views": int(stats.get('viewCount', 0)),
                        "likes": int(stats.get('likeCount', 0)),
                        "comments_count": int(stats.get('commentCount', 0)),
                        "published_at": item['snippet'].get('publishedAt', ''),
                        "embedded_link": f"https://www.youtube.com/embed/{video_id}",
                        **analysis
                    })

                except Exception as e:
                    logger.warning(f"Skipping malformed video item: {str(e)}")
                    continue

            return JsonResponse({
                "status": "success",
                "count": len(video_data),
                "videos": video_data,
                "source": "api"
            })

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return JsonResponse({"error": "Search failed"}, status=500)

    return JsonResponse({"error": "Only POST requests are accepted"}, status=405)


def analyze_video(video_id):
    """Analyze video comments with improved accuracy"""
    try:
        logger.info(f"Starting analysis for video: {video_id}")
        comments = fetch_youtube_comments(video_id, max_results=100)  # Limit to 100 comments
        
        if not comments:
            logger.warning("No comments available for analysis")
            return {
                'positive_comments_count': 0,
                'negative_comments_count': 0,
                'neutral_comments_count': 0,
                'positivity_percentage': 0,
                'comments_analyzed': 0
            }
        
        positive = 0
        neutral = 0
        negative = 0
        
        for comment in comments:
            try:
                text = comment['text'].lower()
                
                # Use both sentiment analysis and keyword matching
                analysis = TextBlob(text)
                polarity = analysis.sentiment.polarity
                
                # Check for positive keywords
                positive_keywords = {
                    'great', 'perfect', 'beautiful', 'thanks', 'thank', 'ideal', 'wow',
                    'tayyara', 'merci', "j'adore", "j'aime", 'شكرا', 'يعطيك الصحة',
                    'عيشك', 'طيارة', 'يرحم والديك', 'بارك الله فيك', 'مرسي', 'bravo'
                }
                
                has_positive_keyword = any(keyword in text for keyword in positive_keywords)
                
                # Classify based on both sentiment and keywords
                if has_positive_keyword or polarity > 0.2:
                    positive += 1
                elif polarity < -0.2:
                    negative += 1
                else:
                    neutral += 1
                    
            except Exception as e:
                logger.warning(f"Error analyzing comment: {str(e)}")
                neutral += 1
        
        total_analyzed = positive + neutral + negative
        positivity_percentage = (positive / total_analyzed) * 100 if total_analyzed > 0 else 0
        
        logger.info(f"Analysis complete. Positive: {positive}, Negative: {negative}, Neutral: {neutral}")
        
        return {
            'positive_comments_count': positive,
            'negative_comments_count': negative,
            'neutral_comments_count': neutral,
            'positivity_percentage': round(positivity_percentage, 2),
            'comments_analyzed': total_analyzed
        }
        
    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}", exc_info=True)
        return {
            'positive_comments_count': 0,
            'negative_comments_count': 0,
            'neutral_comments_count': 0,
            'positivity_percentage': 0,
            'comments_analyzed': 0
        }

@csrf_exempt
def random_videos(request):
    if request.method == "GET":
        try:
            videos = Video.objects.annotate(
                positive_percentage=100.0 * F('positive_comments_count') / F('comments_count')
            ).filter(
                positive_percentage__gte=70,  
                comments_count__gt=0         
            ).order_by('?')[:20]  

            video_data = [{
                "video_id": video.video_id,
                "title": video.title,
                "url": video.url,
                "thumbnail": video.thumbnail,
                "views": video.views,
                "likes": video.likes,
                "comments_count": video.comments_count,
                "positive_comments_count": video.positive_comments_count,
                "positive_percentage": video.positive_percentage,
                "upload_date": video.upload_date.strftime("%Y-%m-%d %H:%M:%S"),
                "embedded_link": f"https://www.youtube.com/embed/{video.video_id}",
            } for video in videos]

            return JsonResponse({"videos": video_data})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=405)

@csrf_exempt
def get_video_comments(request, video_id):
    if request.method == "GET":
        try:
            comments = fetch_youtube_comments(video_id)
            formatted_comments = []
            for comment in comments:
                formatted_comments.append({
                    'text': comment['text'],
                    'author': comment['author'],
                    'published_at': comment['published_at'],
                    'likes': comment['like_count']
                })
            return JsonResponse({"comments": formatted_comments})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
def analyze_comments(request):
    if request.method == "POST" and sentiment_analyzer:
        try:
            data = json.loads(request.body)
            video_id = data.get("video_id")
            
            if not video_id:
                return JsonResponse({"error": "video_id is required"}, status=400)

            comments = Comment.objects.filter(video__video_id=video_id)
            positive_count = 0

            for comment in comments:
                try:
                    result = sentiment_analyzer(comment.text[:512])[0]
                    is_positive = result["label"] == "POSITIVE"
                    if is_positive:
                        positive_count += 1
                    Comment.objects.filter(id=comment.id).update(is_positive=is_positive)
                except Exception as e:
                    continue

            Video.objects.filter(video_id=video_id).update(
                positive_comments_count=positive_count
            )

            return JsonResponse({
                "message": "Analysis complete",
                "positive_comments_count": positive_count
            })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request or sentiment analyzer not available"}, status=400)


# Added part (lel analyse des comments) 
def fetch_youtube_comments(video_id, max_results=100):
    """Fetch comments directly from YouTube API with better error handling"""
    try:
        logger.info(f"Starting comment fetch for video: {video_id}")
        YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
        if not YOUTUBE_API_KEY:
            logger.error("YouTube API key not configured")
            return []

        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        comments = []
        
        try:
            # First check if comments are enabled
            video_response = youtube.videos().list(
                part='statistics',
                id=video_id
            ).execute()
            
            if not video_response.get('items'):
                logger.warning(f"No video found with ID: {video_id}")
                return []
                
            comment_count = int(video_response['items'][0]['statistics'].get('commentCount', 0))
            logger.info(f"Video has {comment_count} comments")
            
            if comment_count == 0:
                return comments
                
            # Fetch comments with retry logic
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries and len(comments) < max_results:
                try:
                    request = youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=min(max_results - len(comments), 100),
                        textFormat='plainText',
                        order='relevance'
                    )
                    
                    if retry_count > 0:
                        logger.info(f"Retry attempt {retry_count} for video {video_id}")
                        time.sleep(1 * retry_count)  # Exponential backoff
                    
                    response = request.execute()
                    logger.info(f"Received {len(response.get('items', []))} comments in this batch")
                    
                    for item in response.get('items', []):
                        try:
                            comment = item['snippet']['topLevelComment']['snippet']
                            comments.append({
                                'text': comment['textDisplay'],
                                'author': comment['authorDisplayName'],
                                'published_at': comment['publishedAt'],
                                'like_count': comment.get('likeCount', 0),
                                'comment_id': item['id']
                            })
                        except KeyError as e:
                            logger.warning(f"Malformed comment skipped: {str(e)}")
                            continue
                    
                    if 'nextPageToken' in response and len(comments) < max_results:
                        request = youtube.commentThreads().list(
                            part='snippet',
                            videoId=video_id,
                            maxResults=min(max_results - len(comments), 100),
                            pageToken=response['nextPageToken'],
                            textFormat='plainText'
                        )
                    else:
                        break
                        
                    retry_count = 0  # Reset retry count after successful request
                    
                except HttpError as e:
                    if e.resp.status == 403:
                        logger.warning(f"Comments disabled for video {video_id}")
                        break
                    elif e.resp.status == 429:
                        retry_count += 1
                        logger.warning(f"Rate limited, retrying... (attempt {retry_count})")
                        if retry_count >= max_retries:
                            logger.error(f"Max retries reached for video {video_id}")
                            break
                        continue
                    else:
                        logger.error(f"YouTube API error: {str(e)}")
                        break
                except Exception as e:
                    logger.error(f"Error fetching comments: {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        break
                    continue
                    
        except Exception as e:
            logger.error(f"Error in comment fetch: {str(e)}")
            return []
            
        logger.info(f"Total comments fetched: {len(comments)}")
        return comments
        
    except Exception as e:
        logger.error(f"Failed to fetch comments: {str(e)}", exc_info=True)
        return []
# ===== New Frontend Views =====
def home(request):
    popular_videos = Video.objects.filter(
        positive_comments_count__gt=0
    ).order_by('-views')[:12]
    
    recent_videos = Video.objects.filter(
        positive_comments_count__gt=0
    ).order_by('-upload_date')[:12]
    
    return render(request, 'videos/home.html', {
        'popular_videos': popular_videos,
        'recent_videos': recent_videos,
        'page_title': 'Positive Coding Community'
    })

def video_list(request):
    query = request.GET.get('q', '')
    sort = request.GET.get('sort', 'recent')
    
    videos = Video.objects.filter(positive_comments_count__gt=0)
    
    if query:
        videos = videos.filter(
            Q(title__icontains=query) |
            Q(description__icontains=query) |
            Q(query__icontains=query)
        )
    
    if sort == 'views':
        videos = videos.order_by('-views')
    elif sort == 'likes':
        videos = videos.order_by('-likes')
    elif sort == 'comments':
        videos = videos.order_by('-positive_comments_count')
    else:
        videos = videos.order_by('-upload_date')
    
    paginator = Paginator(videos, 24)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'videos/list.html', {
        'videos': page_obj,
        'query': query,
        'sort': sort,
        'page_title': 'Browse Videos'
    })

def video_detail(request, video_id):
    video = get_object_or_404(Video, video_id=video_id)
    comments = video.comment_set.filter(is_positive=True).order_by('-created_at')
    
    related_videos = Video.objects.filter(
        positive_comments_count__gt=0
    ).exclude(
        id=video.id
    ).order_by('?')[:4]
    
    return render(request, 'videos/detail.html', {
        'video': video,
        'comments': comments,
        'related_videos': related_videos,
        'page_title': video.title
    })

# Algo videos
@csrf_exempt
def algorithm_videos(request):
    if request.method == "GET":
        try:
            algorithm_keywords = [
                'algorithm', 'algorithme'
            ]
            
            queries = [Q(title__icontains=keyword) for keyword in algorithm_keywords]
            query = queries.pop()
            for item in queries:
                query |= item
            
            videos = Video.objects.annotate(
                positive_percentage=100.0 * F('positive_comments_count') / F('comments_count')
            ).filter(
                query,
                positive_percentage__gte=70,
                comments_count__gt=0
            ).order_by('-views')  
            
            video_data = []
            for video in videos:
                thumbnail_url = str(video.thumbnail) if video.thumbnail else None
                
                video_data.append({
                    "video_id": video.video_id,
                    "title": video.title,
                    "url": video.url,
                    "thumbnail": thumbnail_url,
                    "views": video.views,
                    "likes": video.likes,
                    "comments_count": video.comments_count,
                    "positive_comments_count": video.positive_comments_count,
                    "positive_percentage": video.positive_percentage,
                    "upload_date": video.upload_date.strftime("%Y-%m-%d %H:%M:%S") if video.upload_date else None,
                    "embedded_link": f"https://www.youtube.com/embed/{video.video_id}",
                })
            
            return JsonResponse({
                "status": "success",
                "videos": video_data
            })
            
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    return JsonResponse({
        "status": "error",
        "message": "Invalid request method"
    }, status=405)

# Deep learning
@csrf_exempt
def dl_videos(request):
    if request.method == "GET":
        try:
            dl_keywords = [
                'deep learning', 'neural network', 'machine learning',
                'cnn', 'rnn', 'lstm', 'gan', 'transformer',
                'tensorflow', 'pytorch', 'keras'
            ]
            
            queries = [Q(title__icontains=keyword) for keyword in dl_keywords]
            query = queries.pop()
            for item in queries:
                query |= item
            
            videos = Video.objects.annotate(
                positive_percentage=100.0 * F('positive_comments_count') / F('comments_count')
            ).filter(
                query,
                positive_percentage__gte=70,
                comments_count__gt=0
            ).order_by('-views')  
            
            video_data = []
            for video in videos:
                thumbnail_url = str(video.thumbnail) if video.thumbnail else None
                
                video_data.append({
                    "video_id": video.video_id,
                    "title": video.title,
                    "url": video.url,
                    "thumbnail": thumbnail_url,
                    "views": video.views,
                    "likes": video.likes,
                    "comments_count": video.comments_count,
                    "positive_comments_count": video.positive_comments_count,
                    "positive_percentage": video.positive_percentage,
                    "upload_date": video.upload_date.strftime("%Y-%m-%d %H:%M:%S") if video.upload_date else None,
                    "embedded_link": f"https://www.youtube.com/embed/{video.video_id}",
                })
            
            return JsonResponse({
                "status": "success",
                "videos": video_data
            })
            
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    return JsonResponse({
        "status": "error",
        "message": "Invalid request method"
    }, status=405)

# db
@csrf_exempt
def dbvds(request):
    if request.method == "GET":
        try:
            database_keywords = [
                'database', 'databases', 'SQL', 'MySQL',
                'Oracle', 'PLSQL', 'PostgreSQL', 'NoSQL',
                'MongoDB', 'Redis', 'SQLite'
            ]
            
            queries = [Q(title__icontains=keyword) for keyword in database_keywords]
            query = queries.pop()
            for item in queries:
                query |= item
            
            videos = Video.objects.annotate(
                positive_percentage=100.0 * F('positive_comments_count') / F('comments_count')
            ).filter(
                query,
                positive_percentage__gte=70,
                comments_count__gt=0
            ).order_by('-views')  

            video_data = []
            for video in videos:
                thumbnail_url = str(video.thumbnail) if video.thumbnail else None
                
                video_data.append({
                    "video_id": video.video_id,
                    "title": video.title,
                    "url": video.url,
                    "thumbnail": thumbnail_url,
                    "views": video.views,
                    "likes": video.likes,
                    "comments_count": video.comments_count,
                    "positive_comments_count": video.positive_comments_count,
                    "positive_percentage": video.positive_percentage,
                    "upload_date": video.upload_date.strftime("%Y-%m-%d %H:%M:%S") if video.upload_date else None,
                    "embedded_link": f"https://www.youtube.com/embed/{video.video_id}",
                })
            
            return JsonResponse({
                "status": "success",
                "videos": video_data
            })
            
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    return JsonResponse({
        "status": "error",
        "message": "Invalid request method"
    }, status=405)

# progl
@csrf_exempt
def p(request):
    if request.method == "GET":
        try:
            programming_keywords = [
                'programming', 'ReactJS', 'NodeJS', 'JavaScript',
                'Python', 'Django', 'Java', 'SpringBoot',
                'PHP', 'C', 'C#', 'C++'
            ]
            
            queries = [Q(title__icontains=keyword) for keyword in programming_keywords]
            query = queries.pop()
            for item in queries:
                query |= item
            
            videos = Video.objects.annotate(
                positive_percentage=100.0 * F('positive_comments_count') / F('comments_count')
            ).filter(
                query,
                positive_percentage__gte=70,
                comments_count__gt=0
            ).order_by('-views')  
            
           
            video_data = []
            for video in videos:
                thumbnail_url = str(video.thumbnail) if video.thumbnail else None
                
                video_data.append({
                    "video_id": video.video_id,
                    "title": video.title,
                    "url": video.url,
                    "thumbnail": thumbnail_url,
                    "views": video.views,
                    "likes": video.likes,
                    "comments_count": video.comments_count,
                    "positive_comments_count": video.positive_comments_count,
                    "positive_percentage": video.positive_percentage,
                    "upload_date": video.upload_date.strftime("%Y-%m-%d %H:%M:%S") if video.upload_date else None,
                    "embedded_link": f"https://www.youtube.com/embed/{video.video_id}",
                })
            
            return JsonResponse({
                "status": "success",
                "videos": video_data
            })
            
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)
    
    return JsonResponse({
        "status": "error",
        "message": "Invalid request method"
    }, status=405)


# EXTRACT 

logger = logging.getLogger(__name__)

def extract_video_id(url):
    """
    Extract YouTube video ID from various URL formats
    """
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\/]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_metadata(video_id):
    """
    Fetch basic metadata about a YouTube video
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(
            f'https://www.youtube.com/watch?v={video_id}',
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.find('meta', property='og:title').get('content', 'Unknown Title')
        
        # Extract views (this might not work as YouTube changes their HTML frequently)
        view_count = 0
        view_element = soup.find('meta', itemprop='interactionCount')
        if view_element:
            try:
                view_count = int(view_element.get('content', '0'))
            except:
                pass
        
        return {
            'title': title,
            'views': view_count,
            'likes': 0  
        }
    except Exception as e:
        logger.error(f"Metadata extraction error: {str(e)}")
        return None


def fetch_video_comments(video_id, filter_positive=True):
    """
    Utility function to fetch comments from database
    """
    queryset = Comment.objects.filter(video__video_id=video_id)
    
    if filter_positive:
        queryset = queryset.filter(is_positive=True)
    
    return queryset.order_by('-created_at').values('id', 'text', 'created_at')

 

logger = logging.getLogger(__name__)

@csrf_exempt
def analyze_youtube_video(request):
    if request.method == 'POST':
        try:
            # Parse request data
            data = json.loads(request.body)
            url = data.get('url')
            
            # Define positive keywords (case insensitive)
            POSITIVE_KEYWORDS = {
                'great', 'perfect', 'beautiful', 'thanks', 'thank', 'ideal', 'wow',
                'tayyara', 'merci', "j'adore", "j'aime", 'شكرا', 'يعطيك الصحة',
                'عيشك', 'طيارة', 'يرحم والديك', 'بارك الله فيك', 'مرسي', 'bravo'
            }
            
            # Extract video ID
            video_id = extract_video_id(url)
            if not video_id:
                return JsonResponse({'error': 'Invalid YouTube URL'}, status=400)
            
            # Get video metadata
            metadata = get_video_metadata(video_id)
            if not metadata:
                return JsonResponse({'error': 'Could not fetch video metadata'}, status=400)
            
            # Fetch comments from database
            try:
                comments_queryset = Comment.objects.filter(
                    video__video_id=video_id
                ).order_by('-created_at').values('id', 'text', 'created_at')
                comments = list(comments_queryset)
                
                # Format datetime fields
                for comment in comments:
                    comment['created_at'] = comment['created_at'].strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.error(f"Error fetching comments: {str(e)}")
                comments = []
            
            # Sentiment analysis with keyword checking
            positive_comments = 0
            negative_comments = 0
            neutral_comments = 0
            keyword_matched_comments = 0
            total_comments = len(comments)
            
            analyzed_comments = []
            
            for comment in comments:
                comment_text = comment['text'].lower()
                is_keyword_positive = any(
                    keyword.lower() in comment_text 
                    for keyword in POSITIVE_KEYWORDS
                )
                
                try:
                    analysis = TextBlob(comment['text'])
                    polarity = analysis.sentiment.polarity
                    
                    # Classify based on both sentiment and keywords
                    if is_keyword_positive or polarity > 0.1:
                        positive_comments += 1
                        sentiment = 'positive'
                        if is_keyword_positive:
                            keyword_matched_comments += 1
                    elif polarity < -0.1:
                        negative_comments += 1
                        sentiment = 'negative'
                    else:
                        neutral_comments += 1
                        sentiment = 'neutral'
                    
                    # Add sentiment info to comment
                    comment['sentiment'] = sentiment
                    comment['is_keyword_positive'] = is_keyword_positive
                    analyzed_comments.append(comment)
                    
                except Exception as e:
                    logger.error(f"Error analyzing comment: {str(e)}")
                    neutral_comments += 1
                    comment['sentiment'] = 'neutral'
                    comment['is_keyword_positive'] = False
                    analyzed_comments.append(comment)
            
            # Generate single sentiment chart
            chart_image = None
            try:
                plt.figure(figsize=(8, 6))
                plt.style.use('ggplot')
                
                # Prepare data
                sizes = [max(0, positive_comments), 
                        max(0, negative_comments), 
                        max(0, neutral_comments)]
                labels = ['Positive', 'Negative', 'Neutral']
                colors = ['#4CAF50', '#F44336', '#FFC107']
                
                # Filter out zero values
                filtered_data = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
                
                if filtered_data:
                    sizes_f, labels_f, colors_f = zip(*filtered_data)
                    plt.pie(
                        sizes_f,
                        labels=labels_f,
                        colors=colors_f,
                        autopct=lambda p: f'{p:.1f}%',
                        startangle=90,
                        textprops={'fontsize': 12}
                    )
                    plt.title('Comment Sentiment Analysis', pad=20, fontsize=14)
                else:
                    plt.text(0.5, 0.5, 'No comment data available', 
                            ha='center', va='center', fontsize=12)
                    plt.title('Comment Sentiment Analysis', pad=20, fontsize=14)
                
                plt.tight_layout()
                
                # Save to buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plt.close()
                
                chart_image = base64.b64encode(buffer.read()).decode('utf-8')
            except Exception as e:
                logger.error(f"Error generating chart: {str(e)}")
                if 'plt' in locals():
                    plt.close()
                chart_image = None
            
            # Prepare response
            response_data = {
                'status': 'success',
                'video': {
                    'title': metadata.get('title', 'Unknown Title'),
                    'views': metadata.get('views', 0),
                    'likes': metadata.get('likes', 0),
                    'comments_count': total_comments,
                    'positive_comments_count': positive_comments,
                    'negative_comments_count': negative_comments,
                    'neutral_comments_count': neutral_comments,
                    'keyword_matched_comments': keyword_matched_comments,
                    'embedded_link': f'https://www.youtube.com/embed/{video_id}',
                    'positive_keywords': list(POSITIVE_KEYWORDS)
                },
                'comments': analyzed_comments
            }
            
            if chart_image:
                response_data['video']['sentiment_chart'] = chart_image
            
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

    # chart par
def get_youtube_client():
    """Helper function to create and return YouTube API client"""
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
    if not YOUTUBE_API_KEY:
        raise ValueError("YouTube API key not configured")
    return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

@csrf_exempt
def get_related_videos(request, video_id):
    """Endpoint to get related videos with matching percentages"""
    if request.method == "GET":
        try:
            youtube = get_youtube_client()
            
            # Get target video details
            target_response = youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            ).execute()
            
            if not target_response.get('items'):
                return JsonResponse({'error': 'Video not found'}, status=404)
                
            target_video = target_response['items'][0]
            target_snippet = target_video['snippet']
            target_stats = target_video['statistics']
            
            # Get category title for better matching
            category_id = target_snippet.get('categoryId')
            category_title = ""
            if category_id:
                try:
                    category_response = youtube.videoCategories().list(
                        part='snippet',
                        id=category_id
                    ).execute()
                    if category_response.get('items'):
                        category_title = category_response['items'][0]['snippet']['title']
                except Exception as e:
                    logger.warning(f"Couldn't fetch category title: {str(e)}")
            
            # Search for related videos
            search_response = youtube.search().list(
                part='id,snippet',
                relatedToVideoId=video_id,
                type='video',
                maxResults=15,
                order='relevance'
            ).execute()
            
            # Get details for all related videos
            related_videos = []
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            
            if video_ids:
                videos_response = youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(video_ids)
                ).execute()
                
                for item in videos_response.get('items', []):
                    match_result = calculate_video_match(
                        youtube,
                        target_snippet, 
                        target_stats,
                        item['snippet'], 
                        item['statistics'],
                        category_title
                    )
                    
                    related_videos.append({
                        'video_id': item['id'],
                        'title': item['snippet']['title'],
                        'thumbnail': item['snippet']['thumbnails']['high']['url'],
                        'views': int(item['statistics'].get('viewCount', 0)),
                        'likes': int(item['statistics'].get('likeCount', 0)),
                        'match_percentage': match_result['percentage'],
                        'match_breakdown': match_result['breakdown'],
                        'category': item['snippet'].get('categoryId', '')
                    })
            
            return JsonResponse({
                'related_videos': sorted(related_videos, key=lambda x: x['match_percentage'], reverse=True)[:10]
            })
            
        except HttpError as e:
            logger.error(f"YouTube API error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=e.resp.status)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid method'}, status=405)

def calculate_video_match(youtube, target_snippet, target_stats, related_snippet, related_stats, target_category_title=""):
    """Calculate match percentage between two videos with detailed breakdown"""
    breakdown = {}
    
    # 1. Title Similarity (30% max)
    title_score = calculate_text_similarity(
        target_snippet['title'],
        related_snippet['title']
    ) * 30
    breakdown['title'] = round(title_score, 1)
    
    # 2. Description/Content Similarity (25% max)
    desc_score = calculate_text_similarity(
        target_snippet.get('description', ''),
        related_snippet.get('description', '')
    ) * 25
    breakdown['content'] = round(desc_score, 1)
    
    # 3. Tags Similarity (25% max)
    tags_score = calculate_tags_similarity(
        target_snippet.get('tags', []),
        related_snippet.get('tags', [])
    ) * 25
    breakdown['tags'] = round(tags_score, 1)
    
    # 4. Category Match (20% max)
    category_score = 0
    if target_snippet.get('categoryId') == related_snippet.get('categoryId'):
        category_score = 20
        # Bonus for same category title if available
        if target_category_title:
            try:
                category_response = youtube.videoCategories().list(
                    part='snippet',
                    id=related_snippet.get('categoryId')
                ).execute()
                if category_response.get('items'):
                    related_category_title = category_response['items'][0]['snippet']['title']
                    if related_category_title.lower() == target_category_title.lower():
                        category_score = 25  # Bonus for exact category match
            except Exception as e:
                logger.warning(f"Couldn't fetch related category title: {str(e)}")
    
    breakdown['category'] = category_score
    
    # Calculate total score (sum of all factors)
    total_score = sum(breakdown.values())
    
    # Normalize to 0-100 range
    normalized_score = min(100, max(0, total_score))
    
    return {
        'percentage': round(normalized_score, 1),
        'breakdown': breakdown
    }

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two texts using TF-IDF weighted cosine similarity"""
    if not text1 or not text2:
        return 0
        
    # Tokenize and create frequency dictionaries
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()
    
    if not tokens1 or not tokens2:
        return 0
    
    # Create frequency counters
    freq1 = Counter(tokens1)
    freq2 = Counter(tokens2)
    
    # Get all unique tokens
    all_tokens = set(tokens1) | set(tokens2)
    
    # Calculate TF-IDF vectors
    vector1 = []
    vector2 = []
    
    for token in all_tokens:
        # Term Frequency
        tf1 = freq1.get(token, 0) / len(tokens1)
        tf2 = freq2.get(token, 0) / len(tokens2)
        
        # Inverse Document Frequency (simplified)
        idf = math.log((2 + 1) / (1 + (1 if token in tokens1 and token in tokens2 else 0)))
        
        vector1.append(tf1 * idf)
        vector2.append(tf2 * idf)
    
    # Calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
        
    return dot_product / (magnitude1 * magnitude2)

def calculate_tags_similarity(tags1, tags2):
    """Calculate similarity between two tag sets using Jaccard index"""
    if not tags1 or not tags2:
        return 0
        
    # Convert to lowercase for case-insensitive comparison
    set1 = set(t.lower() for t in tags1)
    set2 = set(t.lower() for t in tags2)
    
    # Jaccard similarity
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0
    
    jaccard_sim = intersection / union
    
    # Additional weight for exact matches
    exact_match_boost = min(1, intersection * 0.1)  # 10% boost per exact match
    
    return min(1, jaccard_sim + exact_match_boost)

@api_view(['GET'])
def get_video_details(request, video_id):
    video = get_object_or_404(Video, video_id=video_id)
    
    similar_videos = VideoSimilarity.objects.filter(video1=video).select_related('video2')[:10]
    
    data = {
        'video': {
            'video_id': video.video_id,
            'title': video.title,
            'thumbnail': video.thumbnail,
            'views': video.views,
            'likes': video.likes,
            'embedded_link': video.embedded_link
        },
        'similar_videos': [
            {
                'video_id': sv.video2.video_id,
                'title': sv.video2.title,
                'thumbnail': sv.video2.thumbnail,
                'similarity_score': sv.similarity_score,
                'views': sv.video2.views,
                'likes': sv.video2.likes
            }
            for sv in similar_videos
        ]
    }
    
    return Response(data)

@api_view(['POST'])
def calculate_similarity(request):
    try:
        data = json.loads(request.body)
        video1_id = data['video1_id']
        video2_id = data['video2_id']
        
        video1 = Video.objects.get(video_id=video1_id)
        video2 = Video.objects.get(video_id=video2_id)
        
        # This is where you'd call your actual similarity calculation function
        # For demo, we'll just use a placeholder value
        similarity_score = 0.8  # Replace with real calculation
        
        similarity, created = VideoSimilarity.objects.get_or_create(
            video1=video1,
            video2=video2,
            defaults={'similarity_score': similarity_score}
        )
        
        return Response({
            'video1_id': video1_id,
            'video2_id': video2_id,
            'similarity_score': similarity_score,
            'created': created
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=400)
    
VIDEOS = [
    {"id": 1, "title": "Learn React in 10 Minutes", "tags": "react,javascript,frontend"},
    {"id": 2, "title": "React Tutorial for Beginners", "tags": "react,guide,frontend"},
    {"id": 3, "title": "Mastering Django APIs", "tags": "django,backend,python"},
    {"id": 4, "title": "React and Django Integration", "tags": "react,django,fullstack"},
]

def get_related_videos(request, video_id):
    base_video = next((v for v in VIDEOS if v["id"] == int(video_id)), None)
    if not base_video:
        return JsonResponse({"error": "Video not found"}, status=404)

    base_text = (base_video["title"] + " " + base_video["tags"]).lower()

    related_videos = []
    for video in VIDEOS:
        if video["id"] == base_video["id"]:
            continue

        compare_text = (video["title"] + " " + video["tags"]).lower()
        ratio = difflib.SequenceMatcher(None, base_text, compare_text).ratio()
        similarity_percentage = round(ratio * 100, 2)

        related_videos.append({
            "id": video["id"],
            "title": video["title"],
            "tags": video["tags"],
            "similarity": similarity_percentage
        })

    return JsonResponse({"related_videos": related_videos})


@csrf_exempt
def similar_videos(request, video_id):
    try:
        base_video = Video.objects.get(video_id=video_id)
        
        # Get similar videos with their similarity scores
        similar_videos = VideoSimilarity.objects.filter(
            video1=base_video
        ).select_related('video2').order_by('-similarity_score')[:10]
        
        # Format response for network graph
        video_data = {
            "video_id": base_video.video_id,
            "title": base_video.title,
            "thumbnail": str(base_video.thumbnail),
            "embedded_link": f"https://www.youtube.com/embed/{base_video.video_id}",
            "similar_videos": [
                {
                    "video_id": sv.video2.video_id,
                    "title": sv.video2.title,
                    "thumbnail": str(sv.video2.thumbnail),
                    "similarity_percentage": round(sv.similarity_score * 100),
                    "embedded_link": f"https://www.youtube.com/embed/{sv.video2.video_id}",
                    "views": sv.video2.views,
                    "likes": sv.video2.likes
                }
                for sv in similar_videos
            ]
        }
        
        return JsonResponse(video_data)
        
    except Video.DoesNotExist:
        return JsonResponse({"error": "Video not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)




@api_view(['GET'])
def get_video_graph_data(request, video_id):
    try:
        video = Video.objects.get(video_id=video_id)
        similar_videos = VideoSimilarity.objects.filter(
            video1=video
        ).select_related('video2').order_by('-similarity_score')[:10]
        
        nodes = [
            {
                "id": video.video_id,
                "label": video.title[:30] + "..." if len(video.title) > 30 else video.title,
                "title": video.title,
                "image": str(video.thumbnail),
                "size": 25,
                "color": "#4361ee"
            }
        ]
        
        edges = []
        
        for sv in similar_videos:
            nodes.append({
                "id": sv.video2.video_id,
                "label": sv.video2.title[:30] + "..." if len(sv.video2.title) > 30 else sv.video2.title,
                "title": sv.video2.title,
                "image": str(sv.video2.thumbnail),
                "size": 20,
                "color": "#3a0ca3"
            })
            
            edges.append({
                "from": video.video_id,
                "to": sv.video2.video_id,
                "label": f"{round(sv.similarity_score * 100)}%",
                "value": sv.similarity_score * 100,
                "color": get_similarity_color(sv.similarity_score)
            })
        
        return Response({
            "nodes": nodes,
            "edges": edges
        })
        
    except Video.DoesNotExist:
        return Response({"error": "Video not found"}, status=404)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

def get_similarity_color(similarity):
    if similarity >= 0.7: return "#4cc9f0"
    if similarity >= 0.5: return "#f8961e"
    return "#f94144"