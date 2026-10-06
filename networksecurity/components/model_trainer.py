import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact
)

from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import (
    load_numpy_array_data,
    save_object
)


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):

        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train):

        try:
            logging.info("Training Random Forest model")

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train, y_train)

            return model

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self):

        try:

            logging.info("Starting model training")

            # Load transformed training data
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )

            # Load transformed testing data
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            logging.info(
                f"Training array shape: {train_arr.shape}"
            )

            logging.info(
                f"Testing array shape: {test_arr.shape}"
            )

            # Separate input features and target column
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            logging.info("Splitting input and target features completed")

            # Train model
            model = self.train_model(
                X_train,
                y_train
            )

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Accuracy
            train_accuracy = accuracy_score(
                y_train,
                y_train_pred
            )

            test_accuracy = accuracy_score(
                y_test,
                y_test_pred
            )

            logging.info(
                f"Training accuracy: {train_accuracy}"
            )

            logging.info(
                f"Testing accuracy: {test_accuracy}"
            )

            print(f"Training Accuracy: {train_accuracy}")
            print(f"Testing Accuracy: {test_accuracy}")

            # Check expected accuracy
            expected_accuracy = (
                self.model_trainer_config.expected_accuracy
            )

            if test_accuracy < expected_accuracy:
                raise Exception(
                    f"Model accuracy {test_accuracy} is less than "
                    f"expected accuracy {expected_accuracy}"
                )

            # Create model directory
            model_dir = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            os.makedirs(
                model_dir,
                exist_ok=True
            )

            # Save trained model
            save_object(
                self.model_trainer_config.trained_model_file_path,
                model
            )

            logging.info(
                f"Model saved at: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            print(
                f"Model saved at: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise NetworkSecurityException(e, sys)