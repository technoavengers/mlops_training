import streamlit as st
import pandas as pd
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model, plot_confusion_matrix
from tracking import track_experiment

st.title("Random Forest Model Trainer")

if "model" not in st.session_state:
    st.session_state.model = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.number_input("Number of Trees (n_estimators)", min_value=10, max_value=500, value=100, step=10)
max_depth = st.sidebar.number_input("Max Depth (None for unlimited)", min_value=1, max_value=50, value=None)
min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=10, value=2)
random_state = st.sidebar.number_input("Random State", min_value=1, max_value=10000, value=42)

if st.sidebar.button("Train Model"):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        try:
            X_train, X_test, y_train, y_test = preprocess_data(data, target_column="survived", random_state=random_state)

            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth if max_depth != 0 else None,
                "min_samples_split": min_samples_split,
                "random_state": random_state
            }

            model = train_model(X_train, y_train, params)

            accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)

            st.session_state.model = model
            st.session_state.metrics = {
                "accuracy": accuracy,
                "classification_report": report,
                "conf_matrix": conf_matrix,
                "params": params
            }

            st.subheader("Model Evaluation Metrics")
            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.text("Classification Report:")
            st.text(report)

            st.subheader("Confusion Matrix")
            plt = plot_confusion_matrix(conf_matrix)
            st.pyplot(plt)

        except ValueError as e:
            st.error(str(e))
    else:
        st.error("Please upload a CSV file to proceed.")

if st.sidebar.button("Track Experiment with MLflow"):
    if st.session_state.model and st.session_state.metrics:
        run_id, experiment_id = track_experiment(
            st.session_state.model,
            st.session_state.metrics["params"],
            st.session_state.metrics,
            tracking_uri="http://localhost:5001",
            experiment_name="Streamlit Titanic Experiment"
        )
        run_url = f"http://localhost:5001/#/experiments/{experiment_id}/runs/{run_id}"
        experiment_url = f"http://localhost:5001/#/experiments/{experiment_id}"

        st.success("Experiment tracked successfully with MLflow!")
        st.markdown(f"[View this run in MLflow]({run_url})")
        st.markdown(f"[View all runs in MLflow Experiment]({experiment_url})")
    else:
        st.error("You need to train the model first before tracking the experiment.")