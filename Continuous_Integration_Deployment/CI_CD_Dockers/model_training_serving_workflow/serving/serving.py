from fastapi import FastAPI
import pandas as pd
import joblib
from Continuous_Integration_Deployment.CI_CD_Dockers.model_training_serving_workflow.scripts.preprocessing import preprocess_data

# Load model and scaler
model_path = "random_forest_model.pkl"
scaler_path = "scaler.pkl"
model = joblib.load(model_path)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Model Serving API"}

@app.post("/predict/")
def predict(data: dict):
    """
    Make predictions using the trained model.
    """
    df = pd.DataFrame([data])
    processed_data = preprocess_data(df, is_training=False, scaler_path=scaler_path)
    predictions = model.predict(processed_data)
    return {"predictions": predictions.tolist()}
