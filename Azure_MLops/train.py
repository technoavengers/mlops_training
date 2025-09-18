# train.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# -------------------
# Parse CLI arguments
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

# -------------------
# Load dataset
# -------------------
df = pd.read_csv(args.data)
X = pd.get_dummies(df.drop(columns=["actual_demand"]), drop_first=True)
y = df["actual_demand"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------
# Start MLflow run
# -------------------
with mlflow.start_run():
    # Train model
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # -------------------
    # Log params & metrics
    # -------------------
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    print(f"✅ Training complete | MAE={mae:.2f}, R²={r2:.2f}")

    # -------------------
    # Log model
    # -------------------
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="model",
        registered_model_name="walmart-rf-model"  # auto-registers in Model Registry
    )
