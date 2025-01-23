import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


def preprocess_data(data):
    """Preprocess the dataset: encode categorical variables."""
    # Identify categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Use Label Encoding for simplicity (replace with OneHotEncoder if needed)
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    return data, label_encoders

def train_and_log():
    # Load data
    data = pd.read_csv("data/current_data.csv")

    # Preprocess data
    data, label_encoders = preprocess_data(data)

    # Separate features and target
    X = data.drop(columns=["target"])  # Replace "target" with your target column name
    y = data["target"]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy}")

    joblib.dump(model, "model/new_model.joblib")
    joblib.dump(label_encoders, "model/new_label_encoders.joblib")

    return accuracy

if __name__ == "__main__":
    train_and_log()
