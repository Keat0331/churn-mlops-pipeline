# 🚀 End-to-End MLOps Pipeline: Customer Churn Prediction

## Overview
A production-grade MLOps pipeline designed to predict customer churn. Unlike standard tutorials, this project simulates a real-world environment by implementing **automated orchestration**, **schema enforcement**, and **deployment simulation**. 

It uses **Apache Airflow** to manage the lifecycle and **MLflow** for experiment tracking and model registry.

## 🛠️ Tech Stack & Architecture
* **Orchestration:** Apache Airflow (Dockerized)
* **Experiment Tracking:** MLflow (Artifacts, Metrics, Signatures)
* **Model Serving:** Scikit-Learn Pipelines (Preprocessing + Inference)
* **Infrastructure:** Docker Compose (Multi-container setup with Postgres backend)
* **Visualization:** Matplotlib & Seaborn (Automated Feature Importance generation)

## 🌟 Key Features
* **Production-Ready Preprocessing:** Implements `sklearn.pipeline` to bake data cleaning (OneHotEncoding, Scaling) into the model artifact. This prevents "training-serving skew."
* **Schema Enforcement:** Uses MLflow signatures to strictly define input types, ensuring the model rejects malformed data in production.
* **Automated Reporting:** Automatically generates and logs "Feature Importance" plots for every training run.
* **Deployment Verification:** Includes a `test_deployment.py` script that simulates a REST API call with raw JSON input to verify the model is ready for live traffic.

## 💻 How to Run
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/churn-mlops-pipeline.git
    cd churn-mlops-pipeline
    ```
2.  **Start the infrastructure:**
    ```bash
    docker-compose up --build
    ```
3.  **Access Dashboards:**
    * **Airflow:** http://localhost:8081 (User: `admin` / Pass: `admin`)
    * **MLflow:** http://localhost:5000
4.  **Trigger the Pipeline:**
    * Enable the `churn_retraining_pipeline` DAG in Airflow.
    * Watch the pipeline ingest data, train the model, and log artifacts to MLflow.

## 🧪 Verifying Deployment
To simulate a live API request, run the test script inside the container:
```bash
# Get the container ID/Name first (usually churn-mlops-pipeline-airflow-run-1)
docker exec -it [CONTAINER_NAME] python /opt/airflow/scripts/test_deployment.py
```

## 📊 Result: Returns a prediction (Churn: Yes/No) using the latest trained model registry.
* Achieved ~80% accuracy on the Telco Churn dataset.
* Full lineage tracking available in MLflow.