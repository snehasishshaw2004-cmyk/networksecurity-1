import os
import sys

import mlflow

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.metric.estimator import NetworkModel

from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)

from networksecurity.utils.ml_utils.metric.classfication_metric import (
    get_classification_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.metrics import accuracy_score


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

    def train_model(
        self,
        X_train,
        y_train,
        x_test,
        y_test
    ):

        try:

            # ==========================================================
            # MLflow Configuration
            # ==========================================================

            mlflow.set_tracking_uri(
                "http://127.0.0.1:5000"
            )

            print("================================")
            print("MLflow Tracking URI:")
            print(mlflow.get_tracking_uri())
            print("================================")

            mlflow.set_experiment(
                "Network Security"
            )

            # ==========================================================
            # Models
            # ==========================================================

            models = {

                "Random Forest":
                    RandomForestClassifier(
                        verbose=1
                    ),

                "Decision Tree":
                    DecisionTreeClassifier(),

                "Gradient Boosting":
                    GradientBoostingClassifier(
                        verbose=1
                    ),

                "Logistic Regression":
                    LogisticRegression(
                        verbose=1
                    ),

                "AdaBoost":
                    AdaBoostClassifier(),

            }

            # ==========================================================
            # Model Parameters
            # ==========================================================

            params = {

                "Decision Tree": {

                    "criterion": [
                        "gini",
                        "entropy",
                        "log_loss"
                    ],
                },

                "Random Forest": {

                    "n_estimators": [
                        8,
                        16,
                        32,
                        128,
                        256
                    ]
                },

                "Gradient Boosting": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],

                    "subsample": [
                        0.6,
                        0.7,
                        0.75,
                        0.85,
                        0.9
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                "Logistic Regression": {},

                "AdaBoost": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.001
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                }
            }

            # ==========================================================
            # Evaluate All Models
            # ==========================================================

            model_report: dict = evaluate_models(

                X_train=X_train,

                y_train=y_train,

                X_test=x_test,

                y_test=y_test,

                models=models,

                param=params
            )

            # ==========================================================
            # Get Best Model Score
            # ==========================================================

            best_model_score = max(
                sorted(model_report.values())
            )

            # ==========================================================
            # Get Best Model Name
            # ==========================================================

            best_model_name = list(
                model_report.keys()
            )[

                list(
                    model_report.values()
                ).index(
                    best_model_score
                )
            ]

            # ==========================================================
            # Get Best Model
            # ==========================================================

            best_model = models[
                best_model_name
            ]

            print("\n==============================")
            print(
                "BEST MODEL:",
                best_model_name
            )

            print(
                "BEST MODEL SCORE:",
                best_model_score
            )

            print("==============================")

            # ==========================================================
            # Start MLflow Run
            # ==========================================================

            with mlflow.start_run():

                # ------------------------------------------------------
                # Log Model Information
                # ------------------------------------------------------

                mlflow.log_param(
                    "best_model",
                    best_model_name
                )

                mlflow.log_metric(
                    "best_model_score",
                    float(best_model_score)
                )

                # ======================================================
                # Training Prediction
                # ======================================================

                y_train_pred = best_model.predict(
                    X_train
                )

                classification_train_metric = (
                    get_classification_score(

                        y_true=y_train,

                        y_pred=y_train_pred
                    )
                )

                # ======================================================
                # Testing Prediction
                # ======================================================

                y_test_pred = best_model.predict(
                    x_test
                )

                classification_test_metric = (
                    get_classification_score(

                        y_true=y_test,

                        y_pred=y_test_pred
                    )
                )

                # ======================================================
                # Calculate Accuracy
                # ======================================================

                train_accuracy = accuracy_score(

                    y_true=y_train,

                    y_pred=y_train_pred
                )

                test_accuracy = accuracy_score(

                    y_true=y_test,

                    y_pred=y_test_pred
                )

                # ======================================================
                # Get F1 Scores
                # ======================================================

                train_f1_score = (
                    classification_train_metric.f1_score
                )

                test_f1_score = (
                    classification_test_metric.f1_score
                )

                # ======================================================
                # Print Metrics
                # ======================================================

                print(
                    "Training Accuracy:",
                    train_accuracy
                )

                print(
                    "Testing Accuracy:",
                    test_accuracy
                )

                print(
                    "Training F1 Score:",
                    train_f1_score
                )

                print(
                    "Testing F1 Score:",
                    test_f1_score
                )

                # ======================================================
                # Log Metrics to MLflow
                # ======================================================

                mlflow.log_metric(
                    "train_accuracy",
                    float(train_accuracy)
                )

                mlflow.log_metric(
                    "test_accuracy",
                    float(test_accuracy)
                )

                mlflow.log_metric(
                    "train_f1_score",
                    float(train_f1_score)
                )

                mlflow.log_metric(
                    "test_f1_score",
                    float(test_f1_score)
                )

                # ======================================================
                # Load Preprocessor
                # ======================================================

                preprocessor = load_object(

                    file_path=(
                        self.data_transformation_artifact
                        .transformed_object_file_path
                    )
                )

                # ======================================================
                # Create Model Directory
                # ======================================================

                model_dir_path = os.path.dirname(

                    self.model_trainer_config
                    .trained_model_file_path
                )

                os.makedirs(

                    model_dir_path,

                    exist_ok=True
                )

                # ======================================================
                # Create Complete Network Model
                # ======================================================

                network_model = NetworkModel(

                    preprocessor=preprocessor,

                    model=best_model
                )

                # ======================================================
                # Save Complete Model
                # ======================================================

                save_object(

                    self.model_trainer_config
                    .trained_model_file_path,

                    obj=network_model
                )

                # ======================================================
                # Create Final Model Directory
                # ======================================================

                os.makedirs(

                    "final_model",

                    exist_ok=True
                )

                # ======================================================
                # Save Model
                # ======================================================

                save_object(

                    "final_model/model.pkl",

                    best_model
                )

                # ======================================================
                # Create Model Trainer Artifact
                # ======================================================

                model_trainer_artifact = ModelTrainerArtifact(

                    trained_model_file_path=(
                        self.model_trainer_config
                        .trained_model_file_path
                    ),

                    train_metric_artifact=(
                        classification_train_metric
                    ),

                    test_metric_artifact=(
                        classification_test_metric
                    )
                )

                logging.info(

                    f"Model trainer artifact: "
                    f"{model_trainer_artifact}"
                )

                # ======================================================
                # Print MLflow Information
                # ======================================================

                print("\n==============================")
                print("MLFLOW METRICS LOGGED")
                print("==============================")

                print(
                    "MLflow Run ID:",
                    mlflow.active_run().info.run_id
                )

                print(
                    "MLflow Experiment:",
                    mlflow.get_experiment(
                        mlflow.active_run().info.experiment_id
                    ).name
                )

                print(
                    "MLflow URI:",
                    mlflow.get_tracking_uri()
                )

                print(
                    "Train Accuracy:",
                    train_accuracy
                )

                print(
                    "Test Accuracy:",
                    test_accuracy
                )

                print(
                    "Train F1 Score:",
                    train_f1_score
                )

                print(
                    "Test F1 Score:",
                    test_f1_score
                )

                print("==============================\n")

                return model_trainer_artifact

        except Exception as e:

            raise NetworkSecurityException(
                e,
                sys
            )

    def initiate_model_trainer(
        self
    ) -> ModelTrainerArtifact:

        try:

            # ==========================================================
            # Get Train File
            # ==========================================================

            train_file_path = (

                self.data_transformation_artifact
                .transformed_train_file_path
            )

            # ==========================================================
            # Get Test File
            # ==========================================================

            test_file_path = (

                self.data_transformation_artifact
                .transformed_test_file_path
            )

            # ==========================================================
            # Load Training Array
            # ==========================================================

            train_arr = load_numpy_array_data(

                train_file_path
            )

            # ==========================================================
            # Load Testing Array
            # ==========================================================

            test_arr = load_numpy_array_data(

                test_file_path
            )

            # ==========================================================
            # Separate X and Y
            # ==========================================================

            x_train = train_arr[:, :-1]

            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]

            y_test = test_arr[:, -1]

            # ==========================================================
            # Train Model
            # ==========================================================

            model_trainer_artifact = self.train_model(

                x_train,

                y_train,

                x_test,

                y_test
            )

            return model_trainer_artifact

        except Exception as e:

            raise NetworkSecurityException(
                e,
                sys
            )