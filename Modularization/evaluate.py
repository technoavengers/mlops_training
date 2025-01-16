from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2

def save_metrics(mae, r2, file="metrics.txt"):
    """Save evaluation metrics to a file."""
    with open(file, "w") as f:
        f.write(f"mean_absolute_error: {mae}\n")
        f.write(f"r2_score: {r2}\n")