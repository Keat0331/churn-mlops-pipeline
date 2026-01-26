import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mlflow.models.signature import infer_signature
import sys

# Paths
TRAIN_PATH = '/opt/airflow/data/processed/train.parquet'
TEST_PATH = '/opt/airflow/data/processed/test.parquet'
RUN_ID_FILE = '/opt/airflow/data/latest_run_id.txt'

# Configuration
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

def get_pipeline(params):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL_COLS)
        ])
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(**params))])

def run():
    print("Loading processed data...")
    X_train = pd.read_parquet(TRAIN_PATH)
    y_train = X_train.pop('Churn')
    X_test = pd.read_parquet(TEST_PATH)
    y_test = X_test.pop('Churn')

    # MLflow Setup
    mlflow.set_experiment("Telco_Churn_Production")
    
    # Best Practice: Dynamic Run Names
    run_name = f"churn_run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"Starting Run: {run.info.run_id}")
        
        # Train
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        pipeline = get_pipeline(params)
        pipeline.fit(X_train, y_train)

        # Evaluate
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Log
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1": f1_score(y_test, preds)})
        
        # Signature
        signature = infer_signature(X_train, preds)
        mlflow.sklearn.log_model(pipeline, "churn_model", signature=signature)

        # SAVE THE RUN ID FOR THE NEXT TASK
        with open(RUN_ID_FILE, "w") as f:
            f.write(run.info.run_id)
        
        print(f"Run {run.info.run_id} complete. Accuracy: {acc}")

if __name__ == "__main__":
    run()