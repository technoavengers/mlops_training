# drift_server.py
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import Gauge, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from evidently.report import Report
from evidently.metrics import DataDriftTable

app = FastAPI()

# Prometheus registry & metrics
registry = CollectorRegistry()
drift_detected = Gauge(
    "data_drift_detected", "Whether drift detected (1=yes, 0=no)", registry=registry
)
share_drifted_features = Gauge(
    "share_drifted_features", "Share of features drifted", registry=registry
)

def calculate_drift():
    # Load data (in real use, fetch from DB or storage)
    reference = pd.read_csv("/app/data/reference_data.csv")
    current = pd.read_csv("/app/data/production_data.csv")

    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference, current_data=current)

    drift_summary = report.as_dict()["metrics"][0]["result"]

    drift_detected.set(1 if drift_summary["dataset_drift"] else 0)
    share_drifted_features.set(drift_summary["share_of_drifted_columns"])

@app.get("/metrics")
def metrics():
    calculate_drift()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
