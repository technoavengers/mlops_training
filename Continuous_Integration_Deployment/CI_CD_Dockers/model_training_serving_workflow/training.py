from preprocessing import preprocess_data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import yaml
import os

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

os.makedirs(os.path.dirname("/tmp/scaler.pkl"), exist_ok=True)
os.makedirs(os.path.dirname("/tmp/random_forest_model.pkl"), exist_ok=True)


def train_model():

    # Load dataset
    data = pd.read_csv("walmart.csv")

    # Preprocess data
    X,y = preprocess_data(data, is_training=True, scaler_path="/tmp/scaler.pkl")

    # Train the model
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    model.fit(X, y)

    # Save the model
    model_path = "/tmp/random_forest_model.pkl"
    joblib.dump(model, model_path)

if __name__ == "__main__":
    train_model()