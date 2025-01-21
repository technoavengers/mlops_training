import pandas as pd
import yaml
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

# Load dataset
data = pd.read_csv("data/sales.csv")
X = data[["amount", "units"]]
y = data["units"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

# Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"]
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Log results with MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Sales Forecasting Experiment")
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

# Save metrics to a text file
with open("metrics.txt", "w") as f:
    f.write(f"mse: {mse}\n")

print(f"Model training completed. MSE: {mse}")
