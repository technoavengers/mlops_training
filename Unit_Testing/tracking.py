import mlflow
import mlflow.sklearn

def track_experiment(model, params, metrics, tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.sklearn.log_model(model, artifact_path="random_forest_model")

        return run.info.run_id, run.info.experiment_id