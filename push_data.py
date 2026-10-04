import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")

client = MongoClient(MONGODB_URL)

db = client["network_security"]
collection = db["phishing_data"]

data = pd.read_csv("data/phisingData.csv")

records = data.to_dict("records")

if records:
    collection.insert_many(records)

print(f"{len(records)} records inserted successfully!")