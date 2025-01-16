import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, target_column, test_size=0.2, random_state=42):
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    X = data.drop(columns=[target_column])
    y = data[target_column]
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test