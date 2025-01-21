import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import yaml

def preprocess_data(data, is_training=True, scaler_path=None):
    """
    Preprocess the data for training or inference.
    """
    # Feature Engineering
    data['transaction_date'] = pd.to_datetime(data['transaction_date'], errors='coerce')
    data['transaction_day'] = data['transaction_date'].dt.day
    data['transaction_month'] = data['transaction_date'].dt.month
    data['transaction_weekday'] = data['transaction_date'].dt.weekday
    data['transaction_year'] = data['transaction_date'].dt.year
    data = data.drop(columns=['transaction_date'])

    # Encode categorical variables
    categorical_columns = [
        'product_name','category', 'store_location', 'payment_method', 'promotion_applied', 
        'promotion_type', 'weather_conditions', 'holiday_indicator', 'weekday', 
        'customer_loyalty_level', 'customer_gender'
    ]
    encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col].astype(str))

    # Standardize features
    if is_training:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        if scaler_path:
            joblib.dump(scaler, scaler_path)  # Save the scaler for inference
    else:
        if scaler_path:
            scaler = joblib.load(scaler_path)
            data = scaler.transform(data)
        else:
            raise ValueError("Scaler path must be provided for inference.")

    return pd.DataFrame(data)
