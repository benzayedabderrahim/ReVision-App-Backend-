from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import django_filters
from django.core.validators import MinValueValidator, MaxValueValidator


class RegistrationCode(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.email} - {self.code}"

class Video(models.Model):
    video_id = models.CharField(max_length=50, unique=True)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    url = models.URLField()
    thumbnail = models.URLField()
    views = models.PositiveIntegerField(default=0)
    likes = models.PositiveIntegerField(default=0)
    comments_count = models.PositiveIntegerField(default=0)
    positive_comments_count = models.PositiveIntegerField(default=0)
    upload_date = models.DateTimeField()
    query = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    tags = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['-upload_date']
    
    def __str__(self):
        return self.title


class UserInteraction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="interactions")
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name="interactions")
    watched = models.BooleanField(default=False)
    liked = models.BooleanField(default=False)
    commented = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)  # Automatically set on creation

    def __str__(self):
        return f"{self.user.username} - {self.video.title}"
    

class Comment(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    text = models.TextField()
    is_positive = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Comment on {self.video.title}"



class VideoFilter(django_filters.FilterSet):
    title = django_filters.CharFilter(lookup_expr='icontains', label='Search by title')
    channel = django_filters.CharFilter(lookup_expr='icontains', label='Search by channel')
    description = django_filters.CharFilter(lookup_expr='icontains', label='Search by description')

    class Meta:
        model = Video
        fields = ['title', 'channel', 'description']



class VideoSimilarity(models.Model):
    source_video = models.ForeignKey(
        'Video',
        on_delete=models.CASCADE,
        related_name='similar_to'
    )
    target_video = models.ForeignKey(
        'Video',
        on_delete=models.CASCADE,
        related_name='similar_from'
    )
    similarity_score = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="How similar these videos are (0-1)"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['source_video', 'target_video']]
        verbose_name_plural = "Video Similarities"
        ordering = ['-similarity_score']

    def __str__(self):
        return f"{self.source_video} ↔ {self.target_video}: {self.similarity_score:.2f}"