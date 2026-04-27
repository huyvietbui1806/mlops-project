# 🛡️ Financial Fraud Detection – End-to-End MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) project for detecting financial fraud in transactional systems.

This project goes beyond model training by focusing on production readiness, scalability, reproducibility, and monitoring.

---

## 📦 Project Structure

---

```
mlops-project/
├── .dvc/                         # DVC metadata and local cache
├── .github/workflows/            # CI / CD / CT automation workflows
│   ├── ci.yaml
│   ├── cd.yaml
│   └── ct.yaml
├── configs/
│   └── model_config.yaml         # Model and training configuration
├── data/
│   ├── csv/                      # Source CSV datasets
│   ├── raw/                      # Raw data
│   ├── processed/                # Processed train / validation / test datasets
│   └── sample/                   # Small sample data for testing or demo
├── deployment/
│   ├── k8s/                      # Kubernetes manifests
│   └── mlflow/                   # MLflow Docker Compose setup
├── models/
│   ├── artifacts/                # Intermediate model artifacts
│   └── trained/                  # Final trained model and inference artifacts
├── notebooks/
│   ├── EDA.ipynb                 # Exploratory data analysis
│   ├── FeatureEngineering.ipynb  # Feature engineering experiments
│   └── Modeling.ipynb            # Model comparison and reporting
├── reports/
│   └── training/
│       ├── baseline_leaderboard.csv
│       └── best_model.json
├── reports_monitoring/           # Generated monitoring and drift reports
├── src/
│   ├── api/                      # FastAPI inference service
│   │   ├── main.py
│   │   ├── inference.py
│   │   ├── schemas.py
│   │   ├── logger.py
│   │   ├── prediction_store.py
│   │   └── feedback_store.py
│   ├── data/
│   │   └── get_data_v1.py
│   ├── features/
│   │   └── FeatureEngineering.py
│   ├── mlops_project/
│   │   └── __init__.py
│   ├── model/
│   │   ├── train_models.py
│   │   └── tune_model.py
│   ├── monitoring/
│   │   ├── metrics.py
│   │   ├── report_store.py
│   │   └── run_drift_report.py
│   └── streamlit/
│       ├── app.py
│       ├── web.py
│       ├── requirements.txt
│       └── css/
├── Dockerfile                    # Docker image for the API service
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # Locked DVC pipeline state
├── params.yaml                   # Main pipeline parameters
├── params.ci.yaml                # Lightweight parameters for CI
├── pyproject.toml                # Project dependencies and configuration
├── uv.lock                       # Reproducible dependency lock file
└── README.md
```
---

## ⚙️ Tech stack

| Category              | Tools / Frameworks                           |
| --------------------- | -------------------------------------------- |
| Language              | Python 3.11                                  |
| Environment Manager   | uv                                           |
| Data Processing       | Pandas, NumPy                                |
| Machine Learning      | scikit-learn, XGBoost, LightGBM, CatBoost    |
| Experiment Tracking   | MLflow                                       |
| Pipeline Management   | DVC                                          |
| API Serving           | FastAPI, Uvicorn                             |
| Web UI                | Streamlit                                    |
| Containerization      | Docker                                       |
| Orchestration         | Kubernetes                                   |
| CI / CD / CT          | GitHub Actions                               |
| Model/Data Drift      | Evidently AI                                 |
| Metrics & Dashboard   | Prometheus, Grafana                          |
---

## 🧪 Environment Setup

1. **Clone repository:**
```
git clone https://github.com/huyvietbui1806/mlops-project.git
cd mlops-project
```

## 2. Create virtual environment

```bash
uv venv .venv --python 3.11
```

Activate the environment:

### Windows PowerShell

```bash
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

3. **Install dependencies:**
```
uv sync
```
## 📊 Setup MLflow for Experiment Tracking
MLflow is configured using Docker Compose inside:
```
deployment/mlflow/docker-compose.yaml
```
**Start MLflow server:**
```
cd deployment/mlflow
docker compose up -d
docker compose ps
```
Access MLflow UI:
```
http://localhost:5555
```
## 🔁 Model Workflow
### 🧹 Step 1: Data & Feature Engineering
Run feature engineering pipeline:
```bash
uv run python src/features/FeatureEngineering.py
```
This step:
- Cleans transactional data
- Creates behavioral features (velocity, ratios, time-based)
- Main outputs:

```text
data/processed/
├── train_log.parquet      # Training features for logistic-style models
├── valid_log.parquet      # Validation features for logistic-style models
├── test_log.parquet       # Test features for logistic-style models
├── train_tree.parquet     # Training features for tree-based models
├── valid_tree.parquet     # Validation features for tree-based models
└── test_tree.parquet      # Test features for tree-based models

models/trained/
└── fe_params.pkl          # Feature engineering parameters used for inference

models/artifacts/
├── onehot_encoder.pkl     # Fitted OneHotEncoder for categorical features
├── scaler.pkl             # Fitted scaler for numerical features
└── label_encoders.pkl     # Fitted label encoders for categorical/tree features
```
### 🧠 Step 2: Model Training
Train models using configuration:
```bash
uv run python src/model/train_models.py --config configs/model_config.yaml
```
This step:
- Trains multiple models (Logistic Regression, Tree-based)
- Logs experiments to MLflow
- Saves artifacts:
```
reports/training/
├── baseline_leaderboard.csv
└── best_model.json
```
### 🔍 Step 3: Model Tuning
```bash
uv run python src/model/tune_model.py
```
Uses:
- Optuna for hyperparameter tuning
- MLflow for experiment tracking
- threshold selection for final fraud classification

Outputs include:
```
models/trained/
├── trained_model.pkl
├── fe_params.pkl
├── model_columns.pkl
├── trained_model_meta.json
└── tuning_pr_auc_by_trial.png
```
## 🚀 Running FastAPI Inference Service
Start API locally:
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
## 🖥️ Streamlit User Interface

A simple Streamlit interface is provided for interacting with the fraud detection API.

Streamlit files are located in: 
```text
src/streamlit/
```

Run the Streamlit app:
```bash
uv run streamlit run src/streamlit/app.py
```
## 🐳 Docker Setup
**Build API image**
```bash
docker build -t fraud-detection-api .
```
**Run container**
```bash
docker run -p 8000:8000 fraud-detection-api
```
## ☸️ Kubernetes Deployment
Manifests are available in:
```
deployment/k8s/
├── deployment.yaml
└── service.yaml
```
### 🚀 Manual Deployment
```
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
```

## 🔄 CI/CD Pipelines
Located in:
```
.github/workflows/
```
Includes:
- `ci.yaml` → CI (Continuous Integration): Performs code quality checks, linting, and unit testing.
- `ct.yaml` → CT (Continuous Training): Automates model retraining workflows when new data becomes available.
- `cd.yaml` → CD (Continuous Deployment): Builds, packages, and deploys the application to a Kubernetes cluster.

## 📈 Monitoring and Drift Reports

Monitoring logic is implemented in:

```
src/monitoring/
├── metrics.py
├── report_store.py
└── run_drift_report.py
```
## 🚀 Continuous Deployment to Google Kubernetes Engine (GKE)

This project leverages Google Kubernetes Engine (GKE) to enable scalable, reliable, and production-grade deployment of the fraud detection API.

## 🌐 Accessing the Service
```
kubectl get services
```

### 📦 Deployment Workflow

The CD pipeline (cd.yaml) automates the following steps:

Build Docker Image
- The application is containerized using the provided Dockerfile.
- Push to Container Registry
- The image is pushed to Google Artifact Registry for versioned storage.
- Authenticate with Google Cloud
- Secure authentication is handled via a service account.
- Deploy to GKE Cluster
- Kubernetes manifests are applied to update the running service.

### ⚙️ Infrastructure Setup (One-Time Configuration)
1. **Create GKE Cluster**
```
gcloud container clusters create fraud-detection-cluster \
  --zone us-central1-a \
  --num-nodes 2
```
2. **Enable Required Services**
```
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com
```
3. **Configure Cluster Access**
```
gcloud container clusters get-credentials fraud-detection-cluster \
  --zone us-central1-a
```
4. **Create Artifact Registry**
```
gcloud artifacts repositories create fraud-repo \
  --repository-format=docker \
  --location=asia-southeast1
```

### 🔐 GitHub Secrets Configuration

To enable secure deployment, the following secrets must be configured in the repository:

- `GCP_PROJECT_ID` – Google Cloud project identifier
- `GCP_SA_KEY` – Service account credentials (JSON)
- `GKE_CLUSTER_NAME` – Target Kubernetes cluster
- `GKE_ZONE` – Cluster zone

## 🧠 Learn More About MLOps

This project demonstrates:
- End-to-end ML pipeline (DVC)
- Experiment tracking (MLflow)
- Containerized deployment (Docker)
- Scalable orchestration (Kubernetes)
- CI/CD automation

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes with a clear message
4. Open a pull request