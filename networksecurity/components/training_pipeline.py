from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig
)

import sys


class TrainPipeline:

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            data_ingestion_config = DataIngestionConfig(
                self.training_pipeline_config
            )

            data_ingestion = DataIngestion(
                data_ingestion_config
            )

            data_ingestion_artifact = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info(
                "Data ingestion completed successfully"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")

            data_ingestion_artifact = (
                self.start_data_ingestion()
            )

            logging.info(
                f"Data ingestion artifact: "
                f"{data_ingestion_artifact}"
            )

            logging.info(
                "Training pipeline completed successfully"
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)