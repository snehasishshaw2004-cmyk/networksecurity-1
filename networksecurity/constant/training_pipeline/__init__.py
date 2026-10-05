import os
import sys
import numpy as np
import pandas as pd


TARGET_COLUMN: str = "Result"


# ==============================
# Training Pipeline Constants
# ==============================

PIPELINE_NAME: str = "network_security"

ARTIFACT_DIR: str = "artifacts"

FILE_NAME: str = "phishingData.csv"

TRAIN_FILE_NAME: str = "train.csv"

TEST_FILE_NAME: str = "test.csv"


# ==============================
# Data Ingestion Constants
# ==============================

DATA_INGESTION_COLLECTION_NAME: str = "phishing_data"

DATA_INGESTION_DATABASE_NAME: str = "network_security"

DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_INGESTION_INGESTED_DIR: str = "ingested"

DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# ==============================
# Data Validation Constants
# ==============================

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_VALID_DIR_NAME: str = "validated"

DATA_VALIDATION_INVALID_DIR_NAME: str = "invalid"

DATA_VALIDATION_DRIFT_REPORT_DIR_NAME: str = "drift_report"

DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


# ==============================
# Schema Constants
# ==============================

SCHEMA_FILE_PATH: str = os.path.join(
    "data_schema",
    "schema.yaml"
)