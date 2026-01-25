from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG('churn_retraining_pipeline',
         default_args=default_args,
         schedule_interval='@daily', # Runs once a day
         catchup=False) as dag:

    # Task 1: Wait for data (Simulated here, usually a Sensor)
    check_data = BashOperator(
        task_id='check_data_exists',
        bash_command='test -f /opt/airflow/data/raw.csv'
    )

    # Task 2: Run the training script
    train_model = BashOperator(
        task_id='train_model',
        bash_command='python /opt/airflow/scripts/train_model.py'
    )

    check_data >> train_model