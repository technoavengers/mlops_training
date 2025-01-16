import mlflow
import mlflow.sklearn

def track_with_mlflow(model, params, mae, r2):
    """Track the experiment using MLflow."""
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
    mlflow.set_experiment("Walmart Demand Forecast Model Experiment")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("mean absolute error", mae)
        mlflow.log_metric("r2 Score", r2)
        mlflow.sklearn.log_model(model, "random_forest_regressor_model")
    mlflow.end_run()