from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime
from scripts.monitor_drift import monitor_drift
from scripts.train_model import train_and_log
from scripts.evaluate_model import evaluate_model
from airflow.operators.empty import EmptyOperator

default_args = {"owner": "airflow", "retries": 1}

# Thresholds for drift and accuracy
drift_threshold = 0.4  # Example threshold for drift
accuracy_drop_threshold = 0.05  

def decide_retrain(**kwargs):
    # Retrieve results from XCom
    drift = kwargs["ti"].xcom_pull(task_ids="monitor_drift")
    deployed_accuracy,current_accuracy  = kwargs["ti"].xcom_pull(task_ids="evaluate_model")

    # Define thresholds
    accuracy_drop_threshold = 0.05 
    significant_drop = deployed_accuracy - current_accuracy > accuracy_drop_threshold

    if drift > drift_threshold or significant_drop:
        return "train_model"
    else:
        return "skip_training"


# Define the DAG
with DAG(
    "continuous_training_pipeline",
    default_args=default_args,
    description="Continuous Training Pipeline",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    # Task to monitor drift
    monitor_drift_task = PythonOperator(
        task_id="monitor_drift",
        python_callable=monitor_drift,
        op_args=["data/reference_data.csv", "data/current_data.csv"],
    )

    # Task to evaluate the model
    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    # Decision task to branch based on drift and accuracy
    decide_retrain_task = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=decide_retrain,
        provide_context=True,
    )

    # Task to train and track the model
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_and_log,
    )

    # Dummy task to skip training
    skip_training_task = EmptyOperator(
        task_id="skip_training",
    )



    # Define task dependencies
    monitor_drift_task >> evaluate_task >> decide_retrain_task
    decide_retrain_task >> train_task
    decide_retrain_task >> skip_training_task