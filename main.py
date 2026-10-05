import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


if __name__ == "__main__":

    try:

        # =====================================================
        # TRAINING PIPELINE CONFIGURATION
        # =====================================================

        trainingpipelineconfig = TrainingPipelineConfig()


        # =====================================================
        # DATA INGESTION
        # =====================================================

        dataingestionconfig = DataIngestionConfig(
            training_pipeline_config=trainingpipelineconfig
        )

        data_ingestion = DataIngestion(
            data_ingestion_config=dataingestionconfig
        )

        logging.info("Initiate the data ingestion")

        dataingestionartifact = data_ingestion.initiate_data_ingestion()

        logging.info("Data ingestion completed")

        print(dataingestionartifact)


        # =====================================================
        # DATA VALIDATION
        # =====================================================

        datavalidationconfig = DataValidationConfig(
            training_pipeline_config=trainingpipelineconfig
        )

        data_validation = DataValidation(
            data_ingestion_artifact=dataingestionartifact,
            data_validation_config=datavalidationconfig
        )

        logging.info("Initiate the data validation")

        datavalidationartifact = (
            data_validation.initiate_data_validation()
        )

        logging.info("Data validation completed")

        print(datavalidationartifact)


    except Exception as e:

        raise NetworkSecurityException(e, sys)