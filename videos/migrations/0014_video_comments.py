# Generated by Django 5.0.6 on 2025-03-13 22:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0013_remove_video_comment_count_delete_comment'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='comments',
            field=models.PositiveIntegerField(default=0),
        ),
    ]
