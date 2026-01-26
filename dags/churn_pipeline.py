from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 0, # Don't retry automatically for ML tasks (debugging is hard)
}

with DAG('churn_modular_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dag:

    # Task 1: Wait for data
    check_data = BashOperator(
        task_id='check_data_exists',
        bash_command='test -f /opt/airflow/data/raw.csv'
    )

    # Task 2: Cleaning & Splitting
    preprocess = BashOperator(
        task_id='preprocess_data',
        bash_command='python /opt/airflow/scripts/preprocess.py'
    )

    # Task 3: Training & Logging
    train = BashOperator(
        task_id='train_model',
        bash_command='python /opt/airflow/scripts/train.py'
    )

    # Task 4: Model Registration (The Gatekeeper)
    register = BashOperator(
        task_id='register_model',
        bash_command='python /opt/airflow/scripts/register.py'
    )

    # Define the Flow
    check_data >> preprocess >> train >> register