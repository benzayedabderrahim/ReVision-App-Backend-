# Generated by Django 5.0.6 on 2025-03-05 19:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('youtube_link', models.URLField()),
                ('channel', models.CharField(max_length=255)),
                ('published_date', models.DateTimeField()),
            ],
        ),
        migrations.DeleteModel(
            name='YouTubeVideo',
        ),
    ]
