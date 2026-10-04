import os
import json
import sys

from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)


import certifi
ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo 

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def cv_to_json_converter(self, file_path):
    try:
        data = pd.read_csv(file_path)
        data.reset_index(drop=True, inplace=True)
        records =list(json.loads(data.T.to_json()).values())
        return records
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
    def insert_data_to_mongodb(self, records,database,collection):
        try:
           self.database=database
           self.collection=collection
           self.records=records

           self.mongo_client=pymongo.MongoClient(MONGODB_URL)
           self.mongo_db=self.mongo_client[self.database]
           self.mongo_collection=self.mongo_db[self.collection]
           self.mongo_collection.insert_many(self.records)
           return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "KRISHAR"
    Collection = "NetworkData"

    networkobj = NetworkDataExtract()

    records = networkobj.cv_to_json_converter(file_path=FILE_PATH)

    print(records)

    no_of_records = networkobj.insert_data_mongodb(
        records,
        DATABASE,
        Collection
    )

    print(no_of_records)


