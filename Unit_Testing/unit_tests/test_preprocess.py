import pytest
import pandas as pd
from preprocess import preprocess_data

def test_preprocess_data():
    # Update test data to include more samples for each class
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": ["A", "B", "C", "D", "E", "F"],
        "survived": [0, 1, 0, 1, 0, 1]  # Ensure at least two samples per class
    })
    X_train, X_test, y_train, y_test = preprocess_data(data, "survived", test_size=0.4)
    assert X_train.shape[0] > 0  # Check training data is not empty
    assert X_test.shape[0] > 0  # Check test data is not empty
    assert len(set(y_test)) == 2  # Check both classes are represented in the test set

def test_missing_target_column():
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": ["A", "B", "C", "D"]
    })
    with pytest.raises(ValueError, match="Target column 'survived' not found in the dataset."):
        preprocess_data(data, "survived")