import argparse
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

def main():
    parser = argparse.ArgumentParser(description='Data preprocessing for Walmart Demand dataset')
    parser.add_argument('--input', type=str, required=True, help='Input file path (CSV)')
    parser.add_argument('--output', type=str, required=True, help='Output file path (CSV)')
    args = parser.parse_args()

    # Load dataset and preprocess
    dataset = load_dataset(args.input)
    
    categorical_columns = [
    'category', 'store_location', 'payment_method', 'promotion_applied', 
    'promotion_type', 'weather_conditions', 'holiday_indicator', 'weekday', 
    'customer_loyalty_level', 'customer_gender'
    ]
    dataset = encode_categorical(dataset, categorical_columns)

    # Save the processed dataset
    dataset.to_csv(args.output, index=False)
    print(f"Preprocessed data saved to {args.output}")

if __name__ == '__main__':
    main()
