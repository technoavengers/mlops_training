import argparse
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2

def main():
    parser = argparse.ArgumentParser(description='Evaluate the trained model')
    parser.add_argument('--model', type=str, required=True, help='Trained model file path (Pickle)')
    parser.add_argument('--test_data', type=str, required=True, help='Test data file path (CSV)')
    parser.add_argument('--output', type=str, required=True, help='Output metrics file path')
    args = parser.parse_args()

    model = joblib.load(args.model)
    test_data = pd.read_csv(args.test_data)

    X_test = test_data.drop(columns=['transaction_id', 'customer_id', 'product_id', 'product_name', 'actual_demand'])
    y_test = test_data['actual_demand']
    mae, r2 = evaluate_model(model, X_test, y_test)

    with open(args.output, 'w') as f:
        f.write(f"mean_absolute_error: {mae}\n")
        f.write(f"r2_score: {r2}\n")

    print(f"Metrics saved to {args.output}")

if __name__ == '__main__':
    main()
