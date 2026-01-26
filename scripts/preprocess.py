import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
RAW_DATA = '/opt/airflow/data/raw.csv'
OUTPUT_DIR = '/opt/airflow/data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_data(df):
    print("--- Starting Data Cleaning ---")
    # Drop ID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Encode Target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    print("--- Data Cleaning Complete ---")
    return df

def run():
    print(f"Loading raw data from {RAW_DATA}")
    df = pd.read_csv(RAW_DATA)
    
    df_clean = clean_data(df)
    
    # Split
    train, test = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['Churn'])
    
    # Save to Parquet (Preserves schema better than CSV)
    print("Saving processed datasets...")
    train.to_parquet(f"{OUTPUT_DIR}/train.parquet", index=False)
    test.to_parquet(f"{OUTPUT_DIR}/test.parquet", index=False)
    print(f"Success! Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run()