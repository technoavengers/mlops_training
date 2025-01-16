import yaml
from data_preprocessing import load_dataset, encode_categorical
from model import train_model
from evaluate import evaluate_model, save_metrics
from tracking import track_with_mlflow
from sklearn.model_selection import train_test_split
import os

# Load hyperparameters
params_path = os.path.join(os.path.dirname(__file__), "params.yaml")
with open(params_path, "r") as f:
    params = yaml.safe_load(f)

data_path = os.path.join(os.path.dirname(__file__), "data/walmart.csv")
# Load and preprocess the dataset
dataset = load_dataset(data_path)
categorical_columns = [
    'category', 'store_location', 'payment_method', 'promotion_applied', 
    'promotion_type', 'weather_conditions', 'holiday_indicator', 'weekday', 
    'customer_loyalty_level', 'customer_gender'
]
dataset = encode_categorical(dataset, categorical_columns)

# Split the data
X = dataset.drop(columns=['transaction_id', 'customer_id', 'product_id', 'product_name', 'actual_demand'])
y = dataset['actual_demand']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

# Train the model
model = train_model(X_train, y_train, params)
# Evaluate the model
mae, r2 = evaluate_model(model, X_test, y_test)
save_metrics(mae, r2)
# Track the experiment
track_with_mlflow(model, params, mae, r2)

print(f"Model trained. MAE: {mae}, R2: {r2}")