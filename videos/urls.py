from django.urls import path
from .views import get_video_graph_data, similar_videos, get_related_videos, get_video_details, calculate_similarity, random_videos, p,analyze_youtube_video, get_video_comments, algorithm_videos, dl_videos, dbvds, search, signup, firebase_login 

urlpatterns = [
    path("api/auth/login/", firebase_login, name="firebase_login"),
    path("api/auth/signup/", signup, name="signup"),
    path("api/videos/", search, name="videos"),
    path("api/videos/random/", random_videos, name="random_videos"),
    path('api/videos/<str:video_id>/comments/', get_video_comments, name='get_video_comments'),
    path('api/algorithm/', algorithm_videos, name='algorithm_videos'),
    path('api/dlvds/', dl_videos, name='dl_videos'),
    path('api/dbvds/', dbvds, name='db_videos'),
    path('api/p/', p, name='programming languages'),
    path('api/analyze-youtube/', analyze_youtube_video, name='analyze_youtube'),
    path('api/videos/<str:video_id>/', get_video_details),
    path('api/calculate-similarity/', calculate_similarity),
    path('api/related_videos/<int:video_id>/', get_related_videos),
    path("api/videos/<str:video_id>/similar/", similar_videos, name="similar_videos"),
    path('api/videos/<str:video_id>/graph/', get_video_graph_data, name='video_graph_data'),



]
