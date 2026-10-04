import os
from dotenv import load_dotenv
from pymongo import MongoClient

uri = "MONGODB_URL=mongodb+srv://shawsnehasish64_db_user:NetworkDB2026Secure@cluster0.5h6pqi0.mongodb.net/?appName=Cluster0"

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

client = MongoClient(MONGODB_URI)

db = client["networksecurity"]

collection = db["phishingData"]

print("MongoDB connected successfully!")