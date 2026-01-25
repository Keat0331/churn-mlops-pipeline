import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os

# --- CONFIGURATION ---
DATA_PATH = '/opt/airflow/data/raw.csv'
TARGET_COL = 'Churn'
ID_COL = 'customerID'
# Define which columns are which type explicitly
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                    'PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 
                    'Contract', 'PaperlessBilling', 'PaymentMethod']

def clean_data(df):
    """
    Performs robust data cleaning specific to the Telco Churn dataset.
    """
    print("--- Starting Data Cleaning ---")
    
    # 1. Drop ID column (it has no predictive power)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # 2. Handle 'TotalCharges' - The famous Telco dataset bug!
    # It contains empty spaces " " for new customers. We coerce them to NaN, then fill with 0.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 3. Encode Target Variable (Yes/No -> 1/0)
    df[TARGET_COL] = df[TARGET_COL].map({'Yes': 1, 'No': 0})
    
    print("--- Data Cleaning Complete ---")
    return df

def get_preprocessor():
    """
    Returns a Scikit-Learn ColumnTransformer.
    This ensures transformations (Scaling, Encoding) are part of the model pipeline.
    """
    # Pipeline for Numerical Features: Standard Scaling
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pipeline for Categorical Features: One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Combine them
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ])
    
    return preprocessor

# --- MAIN EXECUTION ---

# 1. Load Data
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# 2. Clean Data
df_clean = clean_data(df)

# 3. Split Data
X = df_clean.drop(columns=[TARGET_COL])
y = df_clean[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. MLflow Tracking
mlflow.set_experiment("Telco_Churn_Production")

with mlflow.start_run():
    # A. Define the Model Pipeline
    # We wrap the Preprocessor and the Model into one object.
    # This is BEST PRACTICE for deployment (raw data goes in -> prediction comes out).
    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor()),
        ('classifier', RandomForestClassifier(**rf_params))
    ])

    # B. Train
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # C. Evaluate
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    # Calculate Metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    print(f"metrics: Acc={accuracy:.2f}, F1={f1:.2f}, AUC={roc_auc:.2f}")

    # D. Log to MLflow
    # Log Params
    mlflow.log_params(rf_params)
    
    # Log Metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })

    # Log Model with Signature (Crucial for Deployment!)
    # This tells the model registry exactly what input columns to expect.
    signature = infer_signature(X_train, preds)
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="churn_model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )
    
    # E. (Bonus) Log Feature Importance Plot
    # We have to dig into the pipeline to get feature names
    try:
        # Extract feature names from OneHotEncoder
        ohe_cols = pipeline.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(CATEGORICAL_COLS)
        all_features = NUMERIC_COLS + list(ohe_cols)
        
        # Get importances
        importances = pipeline.named_steps['classifier'].feature_importances_
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=all_features)
        plt.title("Feature Importance")
        plt.tight_layout()
        
        # Save and log
        plot_path = "feature_importance.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path) # Clean up
    except Exception as e:
        print(f"Could not log feature importance: {e}")

    print("Run Complete.")