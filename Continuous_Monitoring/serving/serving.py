from fastapi import FastAPI
import pandas as pd
import joblib
from prometheus_client import Summary, Counter, Gauge, start_http_server
from preprocessing import preprocess_data

# Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing a request')
ERROR_COUNT = Counter('error_count', 'Number of failed predictions')
PREDICTIONS_COUNT = Counter('predictions_count', 'Number of predictions made')

# Load model and scaler
model_path = "/app/models/random_forest_model.pkl"
scaler_path = "/app/models/scaler.pkl"
model = joblib.load(model_path)

app = FastAPI()

# Start Prometheus metrics server
start_http_server(8001)  # Expose metrics at port 8001

@app.get("/")
def home():
    return {"message": "Welcome to the Model Serving API"}

@app.post("/predict/")
@REQUEST_TIME.time()  # Measure request processing time
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        processed_data = preprocess_data(df, is_training=False, scaler_path=scaler_path)
        predictions = model.predict(processed_data)
        PREDICTIONS_COUNT.inc()  # Increment prediction count
        return {"predictions": predictions.tolist()}
    except Exception as e:
        ERROR_COUNT.inc()  # Increment error count
        return {"error": str(e)}
