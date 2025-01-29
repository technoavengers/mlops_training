import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Continuous_Integration_Deployment.CI_CD_Dockers.model_training_serving_workflow.scripts.preprocessing import load_dataset, encode_categorical, preprocess_data


@pytest.fixture
def sample_data():
    """Fixture to provide a sample dataset."""
    return pd.DataFrame({
        "product_name": ["Fridge", "TV", "Smartphone"],
        "category": ["Electronics", "Electronics", "Electronics"],
        "store_location": ["Miami, FL", "Dallas, TX", "Los Angeles, CA"],
        "payment_method": ["Credit Card", "Cash", "Digital Wallet"],
        "promotion_applied": [True, True, False],
        "promotion_type": ["None", "Percentage Discount", "None"],
        "weather_conditions": ["Stormy", "Rainy", "Sunny"],
        "holiday_indicator": [False, False, True],
        "weekday": ["Friday", "Monday", "Sunday"],
        "customer_loyalty_level": ["Silver", "Gold", "Bronze"],
        "customer_gender": ["Other", "Other", "Female"],
        "actual_demand": [100, 200, 150],
        "transaction_date": ["2024-03-31", "2024-07-28", "2024-08-15"]
    })



def test_encode_categorical(sample_data):
    """Test the encode_categorical function."""
    categorical_columns = [
        "product_name", "category", "store_location", "payment_method",
        "promotion_applied", "promotion_type", "weather_conditions",
        "holiday_indicator", "weekday", "customer_loyalty_level", "customer_gender"
    ]

    df_encoded = encode_categorical(sample_data, categorical_columns)

    # Check if all categorical columns are encoded
    for col in categorical_columns:
        assert np.issubdtype(df_encoded[col].dtype, np.number)


def test_preprocess_data(sample_data, tmp_path):
    """Test the preprocess_data function."""
    # Save scaler path
    scaler_path = tmp_path / "scaler.pkl"

    # Run preprocessing in training mode
    X, y = preprocess_data(sample_data, is_training=True, scaler_path=scaler_path)

    # Check if target column is separated
    assert "actual_demand" not in X.columns
    assert len(y) == len(sample_data)