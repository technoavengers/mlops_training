import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def monitor_drift(reference_path, current_path):
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)

    # Generate Data Drift Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    drift_report = report.as_dict()

    drift_share = drift_report["metrics"][0]["result"]["drift_share"]
    return drift_share