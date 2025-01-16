from sklearn.ensemble import RandomForestRegressor

def train_model(X_train, y_train, params):
    """Train a Random Forest model."""
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"]
    )
    model.fit(X_train, y_train)
    return model