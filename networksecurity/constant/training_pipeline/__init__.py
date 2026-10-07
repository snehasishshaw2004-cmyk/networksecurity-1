import os
import sys
import numpy as np
import pandas as pd


TARGET_COLUMN: str = "Result"


# ============================================================
# TRAINING PIPELINE CONSTANTS
# ============================================================

PIPELINE_NAME: str = "network_security"

ARTIFACT_DIR: str = "artifacts"

FILE_NAME: str = "phishingData.csv"

TRAIN_FILE_NAME: str = "train.csv"

TEST_FILE_NAME: str = "test.csv"


# ============================================================
# DATA INGESTION CONSTANTS
# ============================================================

DATA_INGESTION_COLLECTION_NAME: str = "phishing_data"

DATA_INGESTION_DATABASE_NAME: str = "network_security"

DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_INGESTION_INGESTED_DIR: str = "ingested"

DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# ============================================================
# DATA VALIDATION CONSTANTS
# ============================================================

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_VALID_DIR_NAME: str = "validated"

DATA_VALIDATION_INVALID_DIR_NAME: str = "invalid"

DATA_VALIDATION_DRIFT_REPORT_DIR_NAME: str = "drift_report"

DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


# ============================================================
# SCHEMA CONSTANTS
# ============================================================

SCHEMA_FILE_PATH: str = os.path.join(
    "data_schema",
    "schema.yaml"
)


# ============================================================
# DATA TRANSFORMATION CONSTANTS
# ============================================================

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"

DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

DATA_TRANSFORMATION_IMPUTER_PARAMS = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform"
}


# ============================================================
# MODEL TRAINER CONSTANTS
# ============================================================

MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

MODEL_FILE_NAME: str = "model.pkl"

MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05


# ============================================================
# SAVED MODEL CONSTANTS
# ============================================================

SAVED_MODEL_DIR: str = "final_model"