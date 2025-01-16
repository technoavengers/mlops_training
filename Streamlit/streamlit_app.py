import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Initialize session state for the model and metrics
if "model" not in st.session_state:
    st.session_state.model = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None

# Streamlit App
st.title("Random Forest Model Trainer with MLflow Experiment Tracking")

# Upload CSV File
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Sidebar for Parameter Configuration
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.number_input("Number of Trees (n_estimators)", min_value=10, max_value=500, value=100, step=10)
max_depth = st.sidebar.number_input("Max Depth (None for unlimited)", min_value=1, max_value=50, value=None)
min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=10, value=2)
random_state = st.sidebar.number_input("Random State", min_value=1, max_value=10000, value=42)

# Button to Train the Model
if st.sidebar.button("Train Model"):
    if uploaded_file is not None:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        # Separate features (X) and target (y)
        if "survived" not in data.columns:
            st.error("Target column 'survived' is not present in the dataset.")
        else:
            X = data.drop(columns=["survived"])
            y = data["survived"]

            # Ensure all categorical features are encoded
            X = pd.get_dummies(X, drop_first=True)

            # Split and Scale the Dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train the Random Forest Model
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth if max_depth != 0 else None,
                "min_samples_split": min_samples_split,
                "random_state": random_state
            }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # Evaluate the Model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Save model and metrics in session state
            st.session_state.model = model
            st.session_state.metrics = {
                "accuracy": accuracy,
                "classification_report": report,
                "conf_matrix": conf_matrix,
                "params": params
            }

            # Display Metrics
            st.subheader("Model Evaluation Metrics")
            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.text("Classification Report:")
            st.text(report)

            # Display Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            st.pyplot(fig)
    else:
        st.error("Please upload a CSV file to proceed.")

# Button to Track the Experiment with MLflow
if st.sidebar.button("Track Experiment with MLflow"):
    if st.session_state.model and st.session_state.metrics:
        # Initialize MLflow Experiment
        mlflow.set_tracking_uri("http://localhost:5001")  # Update with your MLflow server URI
        mlflow.set_experiment("Streamlit Titanic Experiment")

        # Start MLflow Run
        with mlflow.start_run() as run:
            # Log Parameters, Metrics, and Model
            mlflow.log_params(st.session_state.metrics["params"])
            mlflow.log_metric("accuracy", st.session_state.metrics["accuracy"])
            mlflow.sklearn.log_model(st.session_state.model, artifact_path="random_forest_model")

            # Generate MLflow links
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            run_url = f"http://localhost:5001/#/experiments/{experiment_id}/runs/{run_id}"  # Replace localhost if running on a remote MLflow server
            experiment_url = f"http://localhost:5001/#/experiments/{experiment_id}"

            # Display links in Streamlit
            st.success("Experiment tracked successfully with MLflow!")
            st.markdown(f"[View this run in MLflow]({run_url})")
            st.markdown(f"[View all runs in MLflow Experiment]({experiment_url})")
    else:
        st.error("You need to train the model first before tracking the experiment.")
