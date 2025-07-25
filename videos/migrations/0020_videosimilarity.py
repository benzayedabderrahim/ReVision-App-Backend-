# Generated by Django 5.0.6 on 2025-05-04 15:07

import django.core.validators
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0019_alter_comment_options_alter_video_options_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='VideoSimilarity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('similarity_score', models.FloatField(help_text='How similar these videos are (0-1)', validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(1)])),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('source_video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='similar_to', to='videos.video')),
                ('target_video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='similar_from', to='videos.video')),
            ],
            options={
                'verbose_name_plural': 'Video Similarities',
                'ordering': ['-similarity_score'],
                'unique_together': {('source_video', 'target_video')},
            },
        ),
    ]
