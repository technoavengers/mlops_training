import argparse
import mlflow
import mlflow.sklearn
import joblib
import yaml  # Use yaml instead of json

def track_with_mlflow(model, params, mae, r2):
    """Track the experiment using MLflow."""
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
    mlflow.set_experiment("Walmart Experiment")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("mean absolute error", mae)
        mlflow.log_metric("r2 Score", r2)
        mlflow.sklearn.log_model(model, "random_forest_regressor_model")
    mlflow.end_run()

def main():
    parser = argparse.ArgumentParser(description="Track the model using MLflow")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--params", type=str, required=True, help="Path to parameters file")
    parser.add_argument("--metrics", type=str, required=True, help="Metrics file path")
    
    args = parser.parse_args()

    # Load the trained model
    model = joblib.load(args.model)

    # Load the parameters from the YAML file
    with open(args.params, "r") as f:
        params = yaml.safe_load(f)  # Use safe_load to parse YAML

    # Read the metrics
    with open(args.metrics, "r") as f:
        lines = f.readlines()
        mae = float(lines[0].split(":")[1].strip())
        r2 = float(lines[1].split(":")[1].strip())

    # Track the model with MLflow
    track_with_mlflow(model, params, mae, r2)

if __name__ == "__main__":
    main()
