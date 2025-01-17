import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def load_dataset(data):
    """
    Load and preprocess the dataset.

    Args:
        data (str or pd.DataFrame): File path to the dataset or a pandas DataFrame.

    Returns:
        pd.DataFrame: Preprocessed dataset with date features extracted.
    """
    if isinstance(data, str):  # If a file path is provided, load the file
        dataset = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):  # If already a DataFrame, use it directly
        dataset = data
    else:
        raise ValueError("Input must be a file path or a pandas DataFrame")

    # Feature Engineering: Extract date features
    dataset['transaction_date'] = pd.to_datetime(dataset['transaction_date'], errors='coerce')
    dataset['transaction_day'] = dataset['transaction_date'].dt.day
    dataset['transaction_month'] = dataset['transaction_date'].dt.month
    dataset['transaction_weekday'] = dataset['transaction_date'].dt.weekday
    dataset['transaction_year'] = dataset['transaction_date'].dt.year

    return dataset.drop(columns=['transaction_date'])


def encode_categorical(dataset, categorical_columns):
    """
    Encode categorical variables using LabelEncoder.

    Args:
        dataset (pd.DataFrame): Input dataset.
        categorical_columns (list): List of columns to encode.

    Returns:
        pd.DataFrame: Dataset with categorical columns encoded.
    """
    encoder = LabelEncoder()
    for col in categorical_columns:
        dataset[col] = encoder.fit_transform(dataset[col].astype(str))
    return dataset


def preprocess_data(data, is_training=True, scaler_path=None):
    """
    Preprocess the data for training or inference.

    Args:
        data (pd.DataFrame): The input dataset.
        is_training (bool): Flag to indicate if preprocessing is for training or inference.
        scaler_path (str): Path to save/load the scaler.

    Returns:
        pd.DataFrame: Preprocessed dataset (scaled and encoded).
        pd.Series (optional): Target variable if training.
    """
    # Load dataset
    dataset = load_dataset(data)

    # Categorical Columns
    categorical_columns = [
        "product_name","category", "store_location", "payment_method", "promotion_applied",
        "promotion_type", "weather_conditions", "holiday_indicator", "weekday",
        "customer_loyalty_level", "customer_gender"
    ]

    missing_cols = [col for col in categorical_columns if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing: {missing_cols}")
    
    # Handle missing values
    dataset[categorical_columns] = dataset[categorical_columns].fillna("Unknown")
    

    # Encode categorical variables
    dataset = encode_categorical(dataset, categorical_columns)

    # Separate features (X) and target (y) if training
    if is_training and "actual_demand" in dataset.columns:
        X = dataset.drop(columns=["actual_demand"])
        y = dataset["actual_demand"]
    else:
        X = dataset
    
    # Ensure all data is numeric
    dataset = dataset.select_dtypes(include=["number"])

    # Standardize numerical features
    scaler = StandardScaler()

    if is_training:
        # Fit and transform the scaler for training
        X_scaled = scaler.fit_transform(X)
        if scaler_path:
            joblib.dump(scaler, scaler_path)  # Save the scaler for inference
    else:
        # Use the saved scaler for inference
        if scaler_path:
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X)
        else:
            raise ValueError("Scaler path must be provided for inference.")

    # Return scaled data (and target if training)
    if is_training and "actual_demand" in dataset.columns:
        return pd.DataFrame(X_scaled, columns=X.columns), y
    else:
        return pd.DataFrame(X_scaled, columns=X.columns)
