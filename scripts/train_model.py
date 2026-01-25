import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# 1. Load Data
# In a real job, you'd use S3/GCS. For now, we mount local data.
df = pd.read_csv('/opt/airflow/data/raw.csv')

# 2. Simple Preprocessing (Quick & Dirty for demo)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df = pd.get_dummies(df.drop(['customerID', 'Churn'], axis=1), drop_first=True)
df = df.fillna(0) # Handle missing values simply

X = df
y = pd.read_csv('/opt/airflow/data/raw.csv')['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. MLflow Tracking
# Airflow injects the MLFLOW_TRACKING_URI env var, so we don't need to hardcode it.
mlflow.set_experiment("Telco_Churn_Experiment")

with mlflow.start_run():
    # Train
    params = {"C": 0.5, "solver": "liblinear"}
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log to MLflow
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Model trained with Accuracy: {acc}")