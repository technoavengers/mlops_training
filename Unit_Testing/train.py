from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, params):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model