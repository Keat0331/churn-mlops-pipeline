import mlflow
import pandas as pd

# 1. Simulate "Live" Data (Raw, unscaled, categorical strings)
# This mimics a JSON payload from a frontend website
live_data = pd.DataFrame([{
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,  # New customer
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
    'TotalCharges': 29.85 # or " "
}])

# 2. Load the Model from MLflow
# Note: In production, you'd use the "Model Registry" URI models:/ChurnModel/Production
# For now, we load the latest run from your local folder
# You need the RUN_ID from your MLflow UI (e.g., "8d233...")
logged_model = 'runs:/204c7626d6104983a18990a1b73fc1cf/churn_model'

print(f"Loading model from {logged_model}...")
loaded_model = mlflow.pyfunc.load_model(logged_model)

# 3. Predict
# Notice we pass the RAW live_data. The pipeline handles the scaling/encoding!
prediction = loaded_model.predict(live_data)

print(f"Will this customer churn? {'Yes' if prediction[0] == 1 else 'No'}")