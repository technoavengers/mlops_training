import argparse
import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

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

def main():
    parser = argparse.ArgumentParser(description='Train a model for Walmart demand prediction')
    parser.add_argument('--input', type=str, required=True, help='Input file path (CSV)')
    parser.add_argument('--output', type=str, required=True, help='Output model file path (Pickle)')
    parser.add_argument('--params', type=str, required=True, help='Parameters file (YAML or JSON)')
    args = parser.parse_args()

    dataset = pd.read_csv(args.input)
    X_train = dataset.drop(columns=['transaction_id', 'customer_id', 'product_id', 'product_name', 'actual_demand'])
    y_train = dataset['actual_demand']


    # Load parameters from YAML or JSON (using your existing method)
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)  # or json.load()

    model = train_model(X_train, y_train, params)

    joblib.dump(model, args.output)
    print(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()