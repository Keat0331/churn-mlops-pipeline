# 🚀 End-to-End Cloud Native MLOps Pipeline

## Overview
A production-grade MLOps platform designed to predict customer churn. This project demonstrates a complete "Local-to-Cloud" workflow, simulating a real-world enterprise environment.

It features **automated orchestration** (Airflow), **experiment tracking** (MLflow), **real-time model serving** (FastAPI), and **infrastructure as code** (Kubernetes & Docker Compose).

## 🛠️ Tech Stack & Architecture
* **Orchestration:** Apache Airflow (Dockerized with Init Containers)
* **Experiment Tracking:** MLflow (Artifacts, Metrics, Model Registry)
* **Model Serving:** FastAPI (Real-time REST API)
* **Infrastructure:** Kubernetes (Minikube/GKE) & Docker Compose
* **Security:** Environment Variables & Kubernetes Secrets (Base64 Encoded)
* **Backend:** PostgreSQL (Multi-database architecture)

## 🌟 Key Features
* **Microservice Architecture:** Decoupled services (Training, Tracking, Serving) communicating via shared Docker volumes and REST APIs.
* **Real-Time Inference:** A dedicated **FastAPI** container that dynamically loads the latest `@staging` model from the shared volume to serve predictions via a REST endpoint.
* **Modular Pipeline:** Decoupled DAG (Preprocess $\rightarrow$ Train $\rightarrow$ Register) using intermediate Parquet storage for fault tolerance.
* **Production-Ready Preprocessing:** Implements `sklearn.pipeline` to bake data cleaning (OneHotEncoding, Scaling) into the model artifact to prevent training-serving skew.
* **Self-Healing Infrastructure:** Utilizes `fix-permissions` containers and `livenessProbes` to resolve Docker volume permission issues and race conditions automatically.
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
    *(Note: A `fix-permissions` container will run briefly to ensure Airflow and MLflow can share the artifact volume).*

2.  **Access Dashboards:**
    * **Airflow:** http://localhost:8081 (User/Pass from `.env`)
    * **MLflow:** http://localhost:5000
    * **API Docs:** http://localhost:8000/docs

3.  **Trigger the Pipeline:**
    * Enable the `churn_modular_pipeline` DAG in Airflow.
    * Wait for the `train` task to complete. This saves the model to the shared volume.

4.  **Test the API (Real-Time Prediction):**
    * Since the API loads the model on startup, if you trained a new model, restart the API:
      ```bash
      docker-compose restart api
      ```
    * Send a test request:
      ```bash
      curl -X 'POST' 'http://localhost:8000/predict' \
      -H 'Content-Type: application/json' \
      -d '{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
      }'
      ```
      **Response:** `{"churn_prediction": "No"}`

## ☸️ Kubernetes Deployment (Local / Minikube)
This project includes full Kubernetes manifests for deployment on Minikube.

### Prerequisites
If running locally on Minikube, you must build the images **inside** the Minikube environment so the cluster can access them.

1.  **Point your shell to Minikube's Docker daemon:**
    ```bash
    eval $(minikube docker-env)
    ```
2.  **Build the images:**
    ```bash
    docker build -t churn-airflow:v1 -f docker/Dockerfile.airflow .
    docker build -t churn-mlflow:v1 -f docker/Dockerfile.mlflow .
    docker build -t churn-api:v1 -f docker/Dockerfile.api .
    ```

### Deployment Steps
1.  **Apply Secrets & Storage:**
    ```bash
    kubectl apply -f kubernetes/0-secrets.yaml
    kubectl apply -f kubernetes/0-pvc-artifacts.yaml
    ```
1.  **Deploy Database:**
    ```bash
    kubectl apply -f kubernetes/1-postgres.yaml
    ```
2.  **Deploy Services (Tracking, Orchestration, Serving)::**
    ```bash
    kubectl apply -f kubernetes/2-mlflow.yaml
    kubectl apply -f kubernetes/3-airflow.yaml
    kubectl apply -f kubernetes/4-api.yaml
    ```
3.  **Access Services (Port Forwarding):**
    ```bash
    kubectl port-forward svc/airflow 8081:8080
    kubectl port-forward svc/mlflow 5000:5000
    kubectl port-forward svc/api 8000:8000
    ```

## ☁️ Google Kubernetes Engine (GKE) Deployment
This section details how to deploy this project to the cloud (GCP).

### 1. Prerequisites & Setup
* **Google Cloud CLI** installed and authenticated (`gcloud init`).
* **Billing Enabled** on your GCP Project.
* **APIs Enabled:** Kubernetes Engine, Artifact Registry, Compute Engine.

### 2. Create Cloud Infrastructure
```bash
# 1. Create Artifact Registry (to store Docker images)
gcloud artifacts repositories create churn-repo \
    --repository-format=docker \
    --location=asia-southeast1 \
    --description="MLOps Docker Repository"

# 2. Configure Docker Auth
gcloud auth configure-docker asia-southeast1-docker.pkg.dev

# 3. Create GKE Cluster
gcloud container clusters create churn-cluster \
    --zone asia-southeast1-a \
    --num-nodes 1 \
    --machine-type e2-standard-4

# 4. Connect kubectl to the cluster
gcloud container clusters get-credentials churn-cluster --zone asia-southeast1-a

```

### 3. Build & Push Images

Replace `[PROJECT_ID]` with your actual GCP Project ID.

```bash
# Build & Push Airflow
docker build -t asia-southeast1-docker.pkg.dev/[PROJECT_ID]/churn-repo/churn-airflow:v1 -f docker/Dockerfile.airflow .
docker push asia-southeast1-docker.pkg.dev/[PROJECT_ID]/churn-repo/churn-airflow:v1

# Build & Push MLflow
docker build -t asia-southeast1-docker.pkg.dev/[PROJECT_ID]/churn-repo/churn-mlflow:v1 -f docker/Dockerfile.mlflow .
docker push asia-southeast1-docker.pkg.dev/[PROJECT_ID]/churn-repo/churn-mlflow:v1

```

### 4. Update YAML Manifests

Before deploying, you must update the `image:` field in your YAML files to point to the new Artifact Registry URL.

* **`kubernetes/3-airflow.yaml`**: Update **both** the `initContainer` and `main container` image paths.
* **`kubernetes/2-mlflow.yaml`**: Update the container image path.
* **Change Policy:** Ensure `imagePullPolicy: Always` is set.

### 5. IAM Permissions (Critical Step)

Grant the GKE Service Account permission to pull images from Artifact Registry.

```bash
# 1. Get Service Account Email
SA_EMAIL=$(gcloud iam service-accounts list --filter="name:'Compute Engine default service account'" --format="value(email)")

# 2. Grant Reader Role
gcloud projects add-iam-policy-binding [PROJECT_ID] \
    --member=serviceAccount:$SA_EMAIL \
    --role=roles/artifactregistry.reader

```

### 6. Deploy & Access

```bash
# Deploy to GKE
kubectl apply -f kubernetes/

# Access via Port Forwarding (Recommended for Demo)
kubectl port-forward svc/airflow 8081:8080
kubectl port-forward svc/mlflow 5000:5000

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

1. **Copy the updated script:**
(Since the Docker image is immutable, we copy the latest local test script—which uses the registry alias—into the running pod).
```bash
# Get the Airflow pod name
kubectl get pods -l app=airflow

# Copy the script
kubectl cp scripts/test_deployment.py [POD_NAME]:/opt/airflow/scripts/test_deployment.py

```


2. **Execute the Test:**
```bash
kubectl exec -it [POD_NAME] -- python /opt/airflow/scripts/test_deployment.py

```



**Expected Output:**

```text
Loading model from Registry: models:/Telco_Churn_Model@staging...
✅ Model loaded successfully.
Will this customer churn? No

```

## 📊 Results

* Achieved ~81% accuracy on the Telco Churn dataset.
* Full lineage tracking available in MLflow.
* Automated promotion of valid models to the "Staging" registry alias.

```

```