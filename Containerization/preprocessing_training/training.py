from preprocessing import preprocess_data
from tracking import init_mlflow_tracking, log_mlflow_metrics
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import yaml

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

def train_model():
    # Initialize MLflow tracking
    init_mlflow_tracking()

    # Load dataset
    data = pd.read_csv("walmart.csv")

    # Preprocess data
    X,y = preprocess_data(data, is_training=True, scaler_path="/shared/scaler.pkl")

    # Train the model
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    model.fit(X, y)

    # Save the model
    model_path = "/shared/random_forest_model.pkl"
    joblib.dump(model, model_path)

    # Evaluate the model
    mae = 0.5  # Replace with actual evaluation
    r2 = 0.9   # Replace with actual evaluation

    # Log metrics to MLflow
    log_mlflow_metrics(params, {"mae": mae, "r2": r2}, model_path)
    print("Model saved and logged to MLflow.")

if __name__ == "__main__":
    train_model()