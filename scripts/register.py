import mlflow
from mlflow.tracking import MlflowClient

RUN_ID_FILE = '/opt/airflow/data/latest_run_id.txt'
MIN_ACCURACY = 0.75

def run():
    # 1. Read the Run ID passed from the training step
    try:
        with open(RUN_ID_FILE, "r") as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        print("Error: No run ID found. Did training fail?")
        exit(1)

    print(f"Checking model from Run ID: {run_id}")
    
    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    accuracy = metrics.get("accuracy", 0)

    print(f"Model Accuracy: {accuracy}")

    # 2. Gatekeeper Logic
    if accuracy >= MIN_ACCURACY:
        print("✅ Accuracy validation passed. Registering model...")
        
        model_name = "Telco_Churn_Model"
        model_uri = f"runs:/{run_id}/churn_model"
        
        # Register Model
        result = mlflow.register_model(model_uri, model_name)
        
        # Promote to Staging (Best Practice: Use Aliases in MLflow 2.x, or Stages in 1.x)
        # We will set the alias 'Staging' to this version
        client.set_registered_model_alias(model_name, "staging", result.version)
        
        print(f"Model registered as version {result.version} and aliased as 'staging'")
    else:
        print(f"❌ Accuracy {accuracy} is below threshold {MIN_ACCURACY}. Model rejected.")

if __name__ == "__main__":
    run()