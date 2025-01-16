import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn

# Load hyperparameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load the dataset
file_path = 'data/walmart.csv' 
dataset = pd.read_csv(file_path)

# Convert 'transaction_date' to datetime and extract features
dataset['transaction_date'] = pd.to_datetime(dataset['transaction_date'], errors='coerce')
dataset['transaction_day'] = dataset['transaction_date'].dt.day
dataset['transaction_month'] = dataset['transaction_date'].dt.month
dataset['transaction_weekday'] = dataset['transaction_date'].dt.weekday
dataset['transaction_year'] = dataset['transaction_date'].dt.year
dataset = dataset.drop(columns=['transaction_date'])

# Encode categorical variables
categorical_columns = ['category', 'store_location', 'payment_method', 'promotion_applied', 
                       'promotion_type', 'weather_conditions', 'holiday_indicator', 'weekday', 
                       'customer_loyalty_level', 'customer_gender']

encoder = LabelEncoder()
for col in categorical_columns:
    dataset[col] = encoder.fit_transform(dataset[col].astype(str))

# Define features (X) and target (y)
X = dataset.drop(columns=['transaction_id', 'customer_id', 'product_id', 'product_name', 'actual_demand'])
y = dataset['actual_demand']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    min_samples_split=params["min_samples_split"],
    min_samples_leaf=params["min_samples_leaf"],
    random_state=params["random_state"]
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print("Walmart Demand Forecasting model is trained")

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save metrics to 'metrics.txt'
with open("metrics.txt", "w") as f:
    f.write(f"mean_absolute_error: {mae}\n")
    f.write(f"r2_score: {r2}\n")

# Output the results
print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

### MLFLOW tracking
mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
mlflow.set_experiment("Walmart Demand Forecast Model Experiment")

# Start an MLflow run
with mlflow.start_run():

    # Log parameters, metrics, and the model
    mlflow.log_params(params)
    mlflow.log_metric("mean absolute error", mae)
    mlflow.log_metric("r2 Score", r2)
    mlflow.sklearn.log_model(model, "random_forest_regressor_model")

# End MLflow run
mlflow.end_run()