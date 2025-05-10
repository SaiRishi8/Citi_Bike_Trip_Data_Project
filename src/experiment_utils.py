import logging
import os

import mlflow
from mlflow.models import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("citibike_mlflow")

def set_mlflow_tracking():
    """
    Set up MLflow tracking URI from environment variable.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI environment variable not set.")

    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI set to: {uri}")
    return mlflow

def log_citibike_model_to_mlflow(
    model,
    input_data,
    experiment_name="citibike_demand_experiment",
    metric_name="mae",
    model_name="citibike_demand_model",
    params=None,
    score=None,
):
    """
    Log a trained Citi Bike model and its metadata to MLflow.

    Parameters:
    - model: Trained model (e.g., LightGBM, XGBoost, etc.)
    - input_data: Sample DataFrame used to infer model signature
    - experiment_name: Name of the MLflow experiment
    - metric_name: Evaluation metric name to log (e.g., "mae")
    - model_name: Model name for MLflow registry
    - params: Hyperparameters used to train the model
    - score: Metric score (e.g., MAE value)
    """
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment set: {experiment_name}")

        with mlflow.start_run():
            if params:
                mlflow.log_params(params)
                logger.info(f"Logged hyperparameters: {params}")

            if score is not None:
                mlflow.log_metric(metric_name, score)
                logger.info(f"Logged {metric_name}: {score}")

            signature = infer_signature(input_data, model.predict(input_data))
            logger.info("Model signature inferred for logging.")

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model_artifact",
                signature=signature,
                input_example=input_data,
                registered_model_name=model_name,
            )
            logger.info(f"Model successfully logged and registered as: {model_name}")
            return model_info

    except Exception as e:
        logger.error(f"Failed to log model to MLflow: {e}")
        raise
