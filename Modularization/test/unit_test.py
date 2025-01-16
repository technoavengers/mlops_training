import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from unittest.mock import MagicMock

from data_preprocessing import load_dataset, encode_categorical
from model import train_model
from evaluate import evaluate_model, save_metrics
from tracking import track_with_mlflow

@pytest.fixture
def sample_params():
    return {
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }

@pytest.fixture
def sample_dataset():
    data = {
       "transaction_id": range(1, 11),
        "customer_id": range(101, 111),
        "product_id": range(1001, 1011),
        "product_name": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "actual_demand": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        "transaction_date": ["2023-01-01"] * 10,
        "category": ["Electronics"] * 5 + ["Clothing"] * 5,
        "store_location": ["NY", "LA"] * 5,
        "payment_method": ["Credit", "Debit"] * 5,
        "promotion_applied": ["Yes", "No"] * 5,
        "promotion_type": ["Discount", "None"] * 5,
        "weather_conditions": ["Sunny", "Rainy"] * 5,
        "holiday_indicator": ["No", "Yes"] * 5,
        "weekday": ["Monday", "Tuesday"] * 5,
        "customer_loyalty_level": ["Gold", "Silver"] * 5,
        "customer_gender": ["M", "F"] * 5
    }
    return pd.DataFrame(data)

@pytest.fixture
def preprocessed_dataset(sample_dataset):
    dataset = load_dataset(sample_dataset)
    categorical_columns = [
        'category', 'store_location', 'payment_method', 'promotion_applied', 
        'promotion_type', 'weather_conditions', 'holiday_indicator', 'weekday', 
        'customer_loyalty_level', 'customer_gender'
    ]
    return encode_categorical(dataset, categorical_columns)

def test_load_dataset(sample_dataset):
    dataset = load_dataset(sample_dataset)
    assert not dataset.empty
    assert "transaction_day" in dataset.columns
    assert "transaction_month" in dataset.columns

def test_encode_categorical(sample_dataset):
    categorical_columns = ['category', 'store_location']
    encoded_dataset = encode_categorical(sample_dataset, categorical_columns)
    assert encoded_dataset['category'].dtype == 'int64'
    assert encoded_dataset['store_location'].dtype == 'int64'

def test_split_data(preprocessed_dataset, sample_params):
    X = preprocessed_dataset.drop(columns=['transaction_id', 'customer_id', 'product_id', 'product_name', 'actual_demand'])
    y = preprocessed_dataset['actual_demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sample_params['test_size'], random_state=sample_params['random_state'])
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_train_model(preprocessed_dataset, sample_params):
    X = preprocessed_dataset.drop(columns=['transaction_id', 'customer_id', 'product_id', 'product_name', 'actual_demand'])
    y = preprocessed_dataset['actual_demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sample_params['test_size'], random_state=sample_params['random_state'])
    model = train_model(X_train, y_train, sample_params)
    assert isinstance(model, RandomForestRegressor)

def test_evaluate_model(preprocessed_dataset, sample_params):
    X = preprocessed_dataset.drop(columns=['transaction_id', 'customer_id', 'product_id', 'product_name', 'actual_demand'])
    y = preprocessed_dataset['actual_demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sample_params['test_size'], random_state=sample_params['random_state'])
    model = train_model(X_train, y_train, sample_params)
    mae, r2 = evaluate_model(model, X_test, y_test)
    assert mae >= 0
    assert r2 <= 1

def test_save_metrics():
    mae, r2 = 0.5, 0.9
    save_metrics(mae, r2, file="test_metrics.txt")
    with open("test_metrics.txt", "r") as f:
        content = f.read()
    assert "mean_absolute_error: 0.5" in content
    assert "r2_score: 0.9" in content

def test_track_with_mlflow(sample_params):
    mock_model = MagicMock()
    track_with_mlflow(mock_model, sample_params, 0.5, 0.9)
    # No assertions needed as we are testing the execution of the function