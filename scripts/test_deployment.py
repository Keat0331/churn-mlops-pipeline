import mlflow
import pandas as pd
import sys

# 1. Simulate "Live" Data
live_data = pd.DataFrame([{
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}])

# 2. Load from Registry using Alias
# We use the alias 'staging' which we set in register.py
# This ensures we always get the latest approved model without changing code.
model_name = "Telco_Churn_Model"
alias = "staging"

model_uri = f"models:/{model_name}@{alias}"

print(f"Loading model from Registry: {model_uri}...")

try:
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model. Did you run the 'register_model' task successfully?")
    print(f"Error: {e}")
    sys.exit(1)

# 3. Predict
prediction = loaded_model.predict(live_data)
result = 'Yes' if prediction[0] == 1 else 'No'

print(f"Will this customer churn? {result}")