import streamlit as st
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TargetDriftPreset,
    DataQualityPreset
)

# Paths to datasets
REFERENCE_DATA_PATH = "data/reference_data.csv"
CURRENT_DATA_PATH = "data/current_data.csv"

# Utility function to load datasets
@st.cache_data
def load_data():
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    current_data = pd.read_csv(CURRENT_DATA_PATH)
    return reference_data, current_data

# Utility function to generate and display Evidently reports
def generate_report(preset, reference_data, current_data, report_title):
    with st.spinner(f"Generating {report_title}..."):
        report = Report(metrics=[preset])
        column_mapping = ColumnMapping()
        report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
        report.save_html("report.html")
    with open("report.html", "r", encoding="utf-8") as file:
        st.components.v1.html(file.read(), height=800, width=1100,scrolling=True)

# Streamlit app
st.title("Evidently AI Dashboard")
st.sidebar.header("Navigation")

# Navigation options
options = [
    "Overview",
    "Data Drift Report",
    "Target Drift Report",
    "Data Quality Report",
]
choice = st.sidebar.radio("Select a Report", options)

# Load datasets
reference_data, current_data = load_data()

# Generate reports based on user selection
if choice == "Overview":
    st.write("Select a report from the sidebar to generate and view it.")
    st.write("Ensure `reference_data.csv` and `current_data.csv` are present in the application folder.")

elif choice == "Data Drift Report":
    generate_report(DataDriftPreset(), reference_data, current_data, "Data Drift Report")

elif choice == "Target Drift Report":
    generate_report(TargetDriftPreset(), reference_data, current_data, "Target Drift Report")

elif choice == "Data Quality Report":
    generate_report(DataQualityPreset(), reference_data, current_data, "Data Quality Report")