from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API")

# 1. Define Input Schema (Pydantic)
# This enforces types. If a user sends "tenure": "abc", the API rejects it automatically.
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# 2. Global Model Variable
model = None

# 3. Load Model on Startup
@app.on_event("startup")
async def load_model():
    global model
    model_name = "Telco_Churn_Model"
    alias = "staging"
    
    print(f"Attempting to load model: models:/{model_name}@{alias}")
    try:
        # MLflow will fetch the file path from the Tracking Server
        # Then it reads the files from the Shared Volume
        model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Tip: Has the pipeline run yet? Is the shared volume mounted?")

# 4. Prediction Endpoint
@app.post("/predict")
async def predict(data: CustomerData):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Check logs.")
    
    # Convert input JSON -> Pandas DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict
    try:
        prediction = model.predict(input_df)
        result = "Yes" if prediction[0] == 1 else "No"
        return {"churn_prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))