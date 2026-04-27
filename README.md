# 🛡️ Financial Fraud Detection – End-to-End MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) project for detecting financial fraud in transactional systems.

This project goes beyond model training by focusing on production readiness, scalability, reproducibility, and monitoring.

---

## 📦 Project Structure

---

```
mlops-project/
│
├── .dvc/                     # DVC metadata
├── .github/workflows/        # CI/CD pipelines
│   ├── ci.yaml
│   ├── cd.yaml
│   └── ct.yaml
│
├── .venv/
├── configs/                  # Model configuration
│   └── model_config.yaml
│
├── data/
│   ├── csv/                  # Datasets
│   ├── raw/                  # Raw data (DVC tracked)
│   └── sample/               # Sample datasets
│
├── deployment/
│   ├── k8s/                  # Kubernetes manifests
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── mlflow/               # MLflow Docker setup
│       └── docker-compose.yaml
│
├── models/
│   ├── artifacts/            # Encoders, preprocessors
│   └── trained/              # Final trained models
│       ├── fraud_model.pkl
│       ├── fe_params.pkl
│       ├── model_columns.pkl
│       ├── trained_model.pkl
│       └── trained_model_meta.json
│
├── notebooks/                # EDA & experimentation
│   ├── EDA.ipynb
│   ├── FeatureEngineering.ipynb
│   └── Modeling.ipynb
│
├── reports/                  # Training reports
│
├── src/
│   ├── api/                  # FastAPI inference service
│   │   ├── main.py
│   │   ├── inference.py
│   │   └── schemas.py
│   │
│   ├── data/                  
│   ├── features/             # Feature engineering
│   │   └── FeatureEngineering.py
│   │
│   ├── model/                # Training & tuning
│   │   ├── train_models.py
│   │   └── tune_model.py
│   │
│   └── data/                 # Data processing
│
├── Dockerfile                # API container
├── dvc.yaml                  # ML pipeline definition
├── params.yaml               # Pipeline parameters
├── params.ci.yaml            # CI parameters
├── pyproject.toml            # Dependency management
└── README.md
```
---

## ⚙️ Tech stack

| Category      | Tools                                     |
| ------------- | ----------------------------------------- |
| Language      | Python 3.11                              |
| ML Libraries  | scikit-learn, XGBoost, LightGBM, CatBoost |
| Data          | Pandas, NumPy                             |
| Experiment    | MLflow                                    |
| Pipeline      | DVC                                       |
| API           | FastAPI                                   |
| Container     | Docker                                    |
| Orchestration | Kubernetes                                |
| CI/CD         | GitHub Actions                            |

---

## 🧪 Environment Setup

1. **Clone repository:**
```
git clone https://github.com/huyvietbui1806/mlops-project.git
cd mlops-project
```

2. **Create virtual environment:**
```
uv -m venv
uv -m sync
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
```
python src/features/FeatureEngineering.py
```
This step:
- Cleans transactional data
- Creates behavioral features (velocity, ratios, time-based)
- Saves:
    - fe_params.pkl
    - processed dataset

### 🧠 Step 2: Model Training
Train models using configuration:
```
python src/model/train_models.py \
  --config configs/model_config.yaml
```
This step:
- Trains multiple models (Logistic Regression, Tree-based)
- Logs experiments to MLflow
- Saves artifacts:
```
models/trained/
├── trained_model.pkl
├── fraud_model.pkl
├── model_columns.pkl
└── trained_model_meta.json
```
### 🔍 Step 3: Model Tuning
```
python src/model/tune_model.py
```
Uses:
- Optuna for hyperparameter tuning
- MLflow for experiment tracking
- threshold selection for final fraud classification

Outputs include:
- tuned model artifacts
- trained_model_meta.json
- tuning_pr_auc_by_trial.png
- best_model.json
## 🚀 Running FastAPI Inference Service
Start API locally:
```
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
## 🐳 Docker Setup
**Build API image**
```
docker build -t fraud-detection-api .
```
**Run container**
```
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
- `cd.yaml` → CD (Continuous Deployment): Builds, packages, and deploys the application to a Kubernetes cluster.
- `ct.yaml` → CT (Continuous Training): Automates model retraining workflows when new data becomes available.

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

We welcome contributions!
- Fork repository
- Create feature branch
- Commit changes
- Open Pull Request