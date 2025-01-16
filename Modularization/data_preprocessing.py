import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_dataset(data):
    if isinstance(data, str):  # If a file path is provided, load the file
        dataset = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):  # If already a DataFrame, use it directly
        dataset = data
    else:
        raise ValueError("Input must be a file path or a pandas DataFrame")
    """Load and preprocess the dataset."""
    dataset['transaction_date'] = pd.to_datetime(dataset['transaction_date'], errors='coerce')
    dataset['transaction_day'] = dataset['transaction_date'].dt.day
    dataset['transaction_month'] = dataset['transaction_date'].dt.month
    dataset['transaction_weekday'] = dataset['transaction_date'].dt.weekday
    dataset['transaction_year'] = dataset['transaction_date'].dt.year
    return dataset.drop(columns=['transaction_date'])

def encode_categorical(dataset, categorical_columns):
    """Encode categorical variables using LabelEncoder."""
    encoder = LabelEncoder()
    for col in categorical_columns:
        dataset[col] = encoder.fit_transform(dataset[col].astype(str))
    return dataset