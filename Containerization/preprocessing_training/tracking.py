import mlflow
import os

def init_mlflow_tracking():
    """
    Initialize the MLflow tracking server.
    """
    mlflow.set_tracking_uri("http://mlflow_server:5000")  # Use the MLflow server URI
    mlflow.set_experiment("Walmart Sales Prediction Experiment")  # Set the experiment name

def log_mlflow_metrics(params, metrics, model_path):
    """
    Log parameters, metrics, and the model to the MLflow tracking server.
    
    Args:
        params (dict): Hyperparameters of the model.
        metrics (dict): Evaluation metrics (e.g., MAE, R2).
        model_path (str): Path to the saved model file.
    """
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log the model artifact
        mlflow.log_artifact(model_path, artifact_path="models")

        # Get the run information
        run_id = mlflow.active_run().info.run_id
        print(f"Logged run with ID: {run_id}")