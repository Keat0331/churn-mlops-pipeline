# 🚀 End-to-End Cloud Native MLOps Pipeline

## Overview
A production-grade MLOps platform designed to predict customer churn. This project demonstrates a complete "Local-to-Cloud" workflow, simulating a real-world enterprise environment.

It features **automated orchestration** (Airflow), **experiment tracking** (MLflow), and **infrastructure as code** (Kubernetes & Docker Compose), with a strong focus on security (Secret Management) and reliability (Self-Healing Infrastructure).



## 🛠️ Tech Stack & Architecture
* **Orchestration:** Apache Airflow (Dockerized with Init Containers)
* **Experiment Tracking:** MLflow (Artifacts, Metrics, Model Registry)
* **Model Serving:** Scikit-Learn Pipelines (Preprocessing + Inference)
* **Infrastructure:** Kubernetes (Minikube/GKE) & Docker Compose
* **Security:** Environment Variables & Kubernetes Secrets (Base64 Encoded)
* **Backend:** PostgreSQL (Multi-database architecture)

## 🌟 Key Features
* **Hybrid Infrastructure:** Supports both lightweight local development (Docker Compose) and scalable production deployment (Kubernetes).
* **Production-Ready Preprocessing:** Implements `sklearn.pipeline` to bake data cleaning (OneHotEncoding, Scaling) into the model artifact to prevent training-serving skew.
* **Self-Healing Deployments:** Utilizes Kubernetes `livenessProbes` and `initContainers` to resolve race conditions and ensure zero-downtime restarts.
* **Schema Enforcement:** Uses MLflow signatures to strictly define input types, ensuring the model rejects malformed data in production.
* **Automated Reporting:** Automatically generates and logs "Feature Importance" plots for every training run.

## ⚙️ Configuration
**Security Note:** This project uses a `.env` file to manage secrets.
1.  Create a `.env` file in the root directory.
2.  Add the following credentials (do not commit this file to Git):
    ```bash
    POSTGRES_USER=user
    POSTGRES_PASSWORD=password
    POSTGRES_DB=airflow_db
    AIRFLOW_ADMIN_USER=admin
    AIRFLOW_ADMIN_PASS=admin
    AIRFLOW_ADMIN_EMAIL=admin@example.com
    DB_HOST=postgres
    DB_PORT=5432
    ```

## 💻 How to Run (Local Dev)
1.  **Start the infrastructure:**
    ```bash
    docker-compose up --build
    ```
2.  **Access Dashboards:**
    * **Airflow:** http://localhost:8081 (User/Pass from `.env`)
    * **MLflow:** http://localhost:5000
3.  **Trigger the Pipeline:**
    * Enable the `churn_retraining_pipeline` DAG in Airflow.

## ☸️ Kubernetes Deployment (Production)
This project includes full Kubernetes manifests for deployment on GKE or Minikube.
1.  **Apply Secrets & Configs:**
    ```bash
    kubectl apply -f kubernetes/0-secrets.yaml
    kubectl apply -f kubernetes/1-postgres.yaml
    ```
2.  **Deploy Services:**
    ```bash
    kubectl apply -f kubernetes/2-mlflow.yaml
    kubectl apply -f kubernetes/3-airflow.yaml
    ```
3.  **Access Services (NodePort):**
    ```bash
    minikube service airflow --url
    minikube service mlflow --url
    ```

## 🧪 Verifying Deployment
To simulate a live API request, run the test script inside the container:
```bash
# Get the container ID/Name first
docker exec -it [CONTAINER_NAME] python /opt/airflow/scripts/test_deployment.py
```

## 📊 Result: Returns a prediction (Churn: Yes/No) using the latest trained model registry.
* Achieved ~80% accuracy on the Telco Churn dataset.
* Full lineage tracking available in MLflow.