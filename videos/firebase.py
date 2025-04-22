import firebase_admin
from firebase_admin import credentials

if not firebase_admin._apps:
    cred = credentials.Certificate("videos/analyse-project-c2b3e-firebase-adminsdk-fbsvc-e0dcb1dbdf.json")
    firebase_admin.initialize_app(cred)