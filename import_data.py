import pandas as pd
from datetime import datetime
from videos.models import Video

df = pd.read_csv("youtube_education_tunisia.csv")

video_objects = []

for _, row in df.iterrows():
    # Convert the 'Published Date' to a datetime object
    try:
        published_date = datetime.strptime(row["Published Date"], "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        published_date = datetime.strptime(row["Published Date"], "%Y-%m-%dT%H:%M:%SZ")  # Handle different formats
    
    # Create Video instance and append to list
    video_objects.append(
        Video(
            title=row["Title"],
            channel=row["Channel"],
            published_date=published_date,
            youtube_link=row["YouTube Link"],
            description=row["Description"]
        )
    )

Video.objects.bulk_create(video_objects)

print("Data imported into Django database!")
