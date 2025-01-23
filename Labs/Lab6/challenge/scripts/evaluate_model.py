# scripts/evaluate_model.py
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

def preprocess_data(data, encoders):
    for col, encoder in encoders.items():
        if col in data.columns:
            data[col] = encoder.transform(data[col])
    return data


def evaluate_model():
    """
    Evaluate the deployed model on reference and current data.
    Compare performance to decide retraining needs.
    """
    # Load the deployed model
    deployed_model = joblib.load("model/model.joblib")
    label_encoders = joblib.load("model/label_encoders.joblib")

    reference_data = pd.read_csv("data/reference_data.csv")
    X_reference = reference_data.drop(columns=["target"])
    y_reference = reference_data["target"]
    X_reference = preprocess_data(X_reference, label_encoders)

    # Load and preprocess current data
    current_data = pd.read_csv("data/current_data.csv")
    X_current = current_data.drop(columns=["target"])
    y_current = current_data["target"]
    X_current = preprocess_data(X_current, label_encoders)

    # Evaluate deployed model on reference data
    reference_accuracy = accuracy_score(y_reference, deployed_model.predict(X_reference))
    print(f"Deployed Model Accuracy on Reference Data: {reference_accuracy}")

    # Evaluate deployed model on current data
    current_accuracy = accuracy_score(y_current, deployed_model.predict(X_current))
    print(f"Deployed Model Accuracy on Current Data: {current_accuracy}")

    return reference_accuracy,current_accuracy