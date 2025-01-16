from train import train_model
from evaluate import evaluate_model
import pytest
import numpy as np

def test_evaluate_model():
    # Generate synthetic binary classification data
    X_train = np.vstack((np.random.rand(50, 5), np.random.rand(50, 5) + 1))  # Two clusters
    y_train = np.array([0] * 50 + [1] * 50)
    X_test = np.vstack((np.random.rand(10, 5), np.random.rand(10, 5) + 1))  # Two clusters
    y_test = np.array([0] * 10 + [1] * 10)

    # Define the parameters (same as used in actual training)
    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42
    }

    # Train the model using train_model function
    model = train_model(X_train, y_train, params)

    # Evaluate the model
    accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)

    # Assert conditions
    assert accuracy >= 0.7, f"Accuracy {accuracy:.2f} is below the acceptable threshold of 70%"
    assert conf_matrix.shape == (2, 2), "Confusion matrix shape is incorrect"
