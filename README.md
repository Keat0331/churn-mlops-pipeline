# End-to-End MLOps Pipeline: Customer Churn Prediction

## 🚀 Overview
This project implements a fully automated machine learning pipeline for predicting customer churn. It demonstrates a production-ready MLOps architecture using **Apache Airflow** for orchestration, **MLflow** for experiment tracking, and **Docker** for containerized deployment.

## 🛠️ Tech Stack
* **Orchestration:** Apache Airflow
* **Experiment Tracking:** MLflow
* **Containerization:** Docker & Docker Compose
* **Database:** PostgreSQL
* **Language:** Python 3.10

## 🏗️ Architecture
1.  **Data Ingestion:** Simulates daily data arrival.
2.  **Preprocessing:** Cleans categorical and numerical data.
3.  **Model Training:** Trains a Logistic Regression model.
4.  **Evaluation:** Logs accuracy and params to MLflow.
5.  **Registry:** Auto-registers the model if accuracy > threshold.

## 💻 How to Run Locally
1.  Clone the repository:
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/churn-mlops.git
    cd churn-mlops
    ```
2.  Start the services:
    ```bash
    docker-compose up --build
    ```
3.  Access the dashboards:
    * **Airflow:** http://localhost:8081 (User/Pass: `admin`/`admin`)
    * **MLflow:** http://localhost:5000

## 📊 Results
* Achieved ~80% accuracy on the Telco Churn dataset.
* Full lineage tracking available in MLflow.