import json
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

@csrf_exempt
def search(request):
    if request.method == "POST":
        try:
            try:
                data = json.loads(request.body)
                search_query = data.get("prompt", "").strip()
            except json.JSONDecodeError:
                return JsonResponse({"error": "Invalid JSON data"}, status=400)
            
            if not search_query:
                return JsonResponse({"error": "Search query cannot be empty"}, status=400)

            YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
            if not YOUTUBE_API_KEY:
                logger.error("YouTube API key not found in environment variables")
                return JsonResponse({"error": "Service configuration error"}, status=500)

            try:
                youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            except Exception as e:
                logger.error(f"Failed to initialize YouTube client: {str(e)}")
                return JsonResponse({"error": "Service initialization failed"}, status=500)

            try:
                search_response = youtube.search().list(
                    q=search_query,
                    part='id,snippet',
                    maxResults=50,
                    type='video',
                    order='viewCount'
                ).execute()
            except HttpError as e:
                logger.error(f"YouTube API error: {str(e)}")
                return JsonResponse({
                    "error": "YouTube API request failed",
                    "code": e.resp.status
                }, status=502)  
            except Exception as e:
                logger.error(f"Search failed: {str(e)}")
                return JsonResponse({"error": "Search failed"}, status=500)

            video_ids = []
            for item in search_response.get('items', []):
                if 'videoId' in item.get('id', {}):
                    video_ids.append(item['id']['videoId'])

            if not video_ids:
                return JsonResponse({"videos": [], "message": "No videos found"})

            try:
                videos_response = youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(video_ids)
                ).execute()
            except HttpError as e:
                logger.error(f"YouTube API details error: {str(e)}")
                return JsonResponse({"error": "Failed to get video details"}, status=502)

            video_data = []
            for item in videos_response.get('items', []):
                try:
                    video_data.append({
                        "video_id": item['id'],
                        "title": item['snippet'].get('title', 'Untitled'),
                        "description": item['snippet'].get('description', ''),
                        "url": f"https://youtube.com/watch?v={item['id']}",
                        "thumbnail": item['snippet']['thumbnails']['high']['url'],
                        "views": int(item['statistics'].get('viewCount', 0)),
                        "likes": int(item['statistics'].get('likeCount', 0)),
                        "comments_count": int(item['statistics'].get('commentCount', 0)),
                        "published_at": item['snippet'].get('publishedAt', ''),
                        "embedded_link": f"https://www.youtube.com/embed/{item['id']}"
                    })
                except Exception as e:
                    logger.warning(f"Skipping malformed video item: {str(e)}")
                    continue

            return JsonResponse({
                "status": "success",
                "count": len(video_data),
                "videos": video_data
            })

        except Exception as e:
            logger.error(f"Unexpected error in search: {str(e)}", exc_info=True)
            return JsonResponse({
                "error": "An unexpected error occurred",
                "details": str(e)
            }, status=500)

    return JsonResponse({"error": "Only POST requests are accepted"}, status=405)

def analyze_video(video_id):
    try:
        comments = get_video_comments(video_id)  # You'll need to implement this
        
        positive = 0
        neutral = 0
        negative = 0
        
        for comment in comments:
            try:
                analysis = TextBlob(comment['text'])
                polarity = analysis.sentiment.polarity
                
                if polarity > 0.1:
                    positive += 1
                elif polarity < -0.1:
                    negative += 1
                else:
                    neutral += 1
            except:
                neutral += 1
        
        total = positive + neutral + negative
        sentiment_score = (positive - negative) / total if total > 0 else 0
        
        return {
            'positive_comments_count': positive,
            'negative_comments_count': negative,
            'neutral_comments_count': neutral,
            'sentiment_score': round(sentiment_score, 2),
            'comments_analyzed': total
        }
        
    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}")
        return {
            'positive_comments_count': 0,
            'negative_comments_count': 0,
            'neutral_comments_count': 0,
            'sentiment_score': 0,
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
            comments = Comment.objects.filter(
                video__video_id=video_id,
                is_positive=True
            ).order_by('-created_at').values(
                'id',
                'text',
                'created_at'
            )

            comments_list = list(comments)
            for comment in comments_list:
                comment['created_at'] = comment['created_at'].strftime("%Y-%m-%d %H:%M:%S")

            return JsonResponse({"comments": comments_list})
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