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
* **Modular Pipeline Architecture:** Decoupled DAG (Preprocess $\rightarrow$ Train $\rightarrow$ Register) using intermediate Parquet storage. This ensures fault tolerance and easier debugging.
* **Hybrid Infrastructure:** Supports both lightweight local development (Docker Compose) and scalable production deployment (Kubernetes).
* **Production-Ready Preprocessing:** Implements `sklearn.pipeline` to bake data cleaning (OneHotEncoding, Scaling) into the model artifact to prevent training-serving skew.
* **Self-Healing Deployments:** Utilizes Kubernetes `livenessProbes` and `initContainers` to resolve race conditions and ensure zero-downtime restarts.
* **Schema Enforcement:** Uses MLflow signatures to strictly define input types, ensuring the model rejects malformed data in production.
* **Automated Governance:** Implements a "Gatekeeper" script that only registers models to the **Staging** alias if they meet accuracy thresholds.

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
    * Enable the `churn_modular_pipeline` DAG in Airflow.
    * Watch the tasks flow from Data Ingestion $\rightarrow$ Training $\rightarrow$ Model Registration.

## ☸️ Kubernetes Deployment (Production)

This project includes full Kubernetes manifests for deployment on GKE or Minikube.

### Prerequisites (Minikube Only)
If running locally on Minikube, you must build the images **inside** the Minikube environment so the cluster can access them.

1.  **Point your shell to Minikube's Docker daemon:**
    ```bash
    eval $(minikube docker-env)
    ```
2.  **Build the production images:**
    ```bash
    docker build -t churn-airflow:v1 -f docker/Dockerfile.airflow .
    docker build -t churn-mlflow:v1 -f docker/Dockerfile.mlflow .
    ```

### Deployment Steps
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
    *Note: If updating code, run `kubectl delete pods -l app=airflow` to force a restart with the new image.*

3.  **Access Services:**
    ```bash
    minikube service airflow --url
    minikube service mlflow --url
    ```

## 🧪 Verifying Deployment (Continuous Delivery)
This project uses **MLflow Model Registry Aliases**. The test script automatically loads the model tagged as `@staging`, eliminating the need to hardcode Run IDs.

### Option A: Local (Docker Compose)
To simulate a live API request in your local environment:
```bash
# 1. Get the container name
docker ps --filter "name=airflow-run"

# 2. Run the test script inside the container
docker exec -it [CONTAINER_NAME] python /opt/airflow/scripts/test_deployment.py
```

### Option B: Production (Kubernetes)
To verify the model inside the Kubernetes cluster:

1. Copy the updated script: (Since the Docker image is immutable, we copy the latest local test script—which uses the registry alias—into the running pod).
```bash
# Get the Airflow pod name
kubectl get pods -l app=airflow

# Copy the script
kubectl cp scripts/test_deployment.py [POD_NAME]:/opt/airflow/scripts/test_deployment.py
```
2. Execute the Test
```bash
kubectl exec -it [POD_NAME] -- python /opt/airflow/scripts/test_deployment.py
```

## 📊 Result: Returns a prediction (Churn: Yes/No) using the latest trained model registry.
* Achieved ~81% accuracy on the Telco Churn dataset.
* Full lineage tracking available in MLflow.
* Automated promotion of valid models to the "Staging" registry alias.