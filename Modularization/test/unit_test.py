import pytest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.preprocessing import LabelEncoder


# Mock functions based on your original code structure
def load_dataset(data):
    if isinstance(data, str):  # If a file path is provided, load the file
        dataset = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):  # If already a DataFrame, use it directly
        dataset = data
    else:
        raise ValueError("Input must be a file path or a pandas DataFrame")
    dataset['transaction_date'] = pd.to_datetime(dataset['transaction_date'], errors='coerce')
    dataset['transaction_day'] = dataset['transaction_date'].dt.day
    dataset['transaction_month'] = dataset['transaction_date'].dt.month
    dataset['transaction_weekday'] = dataset['transaction_date'].dt.weekday
    dataset['transaction_year'] = dataset['transaction_date'].dt.year
    return dataset.drop(columns=['transaction_date'])

def encode_categorical(dataset, categorical_columns):
    encoder = LabelEncoder()
    for col in categorical_columns:
        dataset[col] = encoder.fit_transform(dataset[col].astype(str))
    return dataset

def train_model(X_train, y_train, params):
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"]
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2

# Testing data preprocessing
class TestDataPreprocessing:

    def test_load_dataset_valid(self):
        data = pd.DataFrame({
            'transaction_date': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'other_column': [1, 2, 3]
        })
        processed_data = load_dataset(data)
        assert 'transaction_day' in processed_data.columns
        assert 'transaction_month' in processed_data.columns

    def test_encode_categorical_valid(self):
        data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C']
        })
        encoded_data = encode_categorical(data, ['category'])
        assert 'category' in encoded_data.columns
        assert encoded_data['category'].iloc[0] == 0  # Label encoded value

# Testing model training
class TestModelTraining:

    def test_train_model(self):
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40]
        })
        y_train = [0, 1, 0, 1]
        params = {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        }
        model = train_model(X_train, y_train, params)
        assert isinstance(model, RandomForestRegressor)

    def test_model_saving(self):
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40]
        })
        y_train = [0, 1, 0, 1]
        params = {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        }
        model = train_model(X_train, y_train, params)
        joblib.dump(model, 'model/random_forest_model.pkl')
        assert os.path.exists('model/random_forest_model.pkl')

# Testing model evaluation
class TestModelEvaluation:

    def test_evaluate_model(self):
        X_test = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [10, 20]
        })
        y_test = [0, 1]
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_test, y_test)
        mae, r2 = evaluate_model(model, X_test, y_test)
        assert isinstance(mae, float)
        assert isinstance(r2, float)


