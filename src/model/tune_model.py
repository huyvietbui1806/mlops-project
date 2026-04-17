from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import optuna
import pandas as pd
import yaml

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =====================
# ARGPARSE
# =====================
def parse_args() -> argparse.Namespace:
    """
    Defaults so you can run:
      python tune_model.py
    """
    parser = argparse.ArgumentParser()

    # Optional config (fallback to defaults in code if missing)
    parser.add_argument("--config", default=None)

    # Optional best model selector
    parser.add_argument("--best-model-json", default="reports/training/best_model.json")

    # FeatureEngineering outputs
    parser.add_argument("--train-log", default="data/processed/train_log.parquet")
    parser.add_argument("--test-log", default="data/processed/test_log.parquet")
    parser.add_argument("--train-tree", default="data/processed/train_tree.parquet")
    parser.add_argument("--test-tree", default="data/processed/test_tree.parquet")

    # FeatureEngineering params
    parser.add_argument("--fe-params", default="models/trained/fe_params.pkl")

    # Output root
    parser.add_argument("--models-dir", default="models")

    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5555")
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--cv", type=int, default=2)

    return parser.parse_args()


# =====================
# UTILS
# =====================
def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}\n"
            f"Tip: run FeatureEngineering.py first or pass correct paths via CLI."
        )

    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)

    raise ValueError(f"Unsupported format: {p.suffix}")


def split_xy(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Columns: {list(df.columns)[:30]} ...")
    return df.drop(columns=[target]), df[target].astype(int)


def maybe_smote(x: pd.DataFrame, y: pd.Series, use_smote: bool, random_state: int):
    if not use_smote:
        return x, y

    smote = SMOTE(random_state=random_state)
    x_res, y_res = smote.fit_resample(x, y)
    return pd.DataFrame(x_res, columns=x.columns), pd.Series(y_res)


def default_cfg() -> dict[str, Any]:
    return {
        "experiment": {"target": "is_fraud", "name": "fraud-model-selection"},
        "resampling": {"use_smote": True, "random_state": 42},
        "models": {
            "logistic_regression": {"dataset": "log", "params": {"solver": "lbfgs", "max_iter": 1000}},
            "xgboost": {"dataset": "tree", "params": {"eval_metric": "logloss", "n_jobs": -1}},
            "lightgbm": {"dataset": "tree", "params": {"n_jobs": -1}},
            "catboost": {"dataset": "tree", "params": {"verbose": False}},
        },
    }


def drop_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMOTE / sklearn cannot handle datetime64 columns.
    Drop all datetime-like columns to avoid DTypePromotionError.
    """
    df = df.copy()

    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if dt_cols:
        logger.warning(f"Dropping datetime columns before SMOTE/training: {dt_cols}")
        df = df.drop(columns=dt_cols)

    # Also: sometimes timestamp was read as object; try best-effort parse then drop if it becomes datetime
    if "timestamp" in df.columns and df["timestamp"].dtype == "object":
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed.notna().any():
            logger.warning("Dropping 'timestamp' column (parsed as datetime) before SMOTE/training.")
            df = df.drop(columns=["timestamp"])

    return df


# =====================
# MODEL
# =====================
def build_model(name: str, params: dict[str, Any]):
    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "xgboost":
        return XGBClassifier(**params)
    if name == "lightgbm":
        return LGBMClassifier(**params)
    if name == "catboost":
        return CatBoostClassifier(**params)
    raise ValueError(f"Unknown model: {name}")


def suggest_params(trial: optuna.Trial, name: str, base: dict[str, Any]) -> dict[str, Any]:
    params = dict(base)

    if name == "logistic_regression":
        params.update(
            {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 1000),
            }
        )
    elif name == "xgboost":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            }
        )
    elif name == "lightgbm":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500),
                "num_leaves": trial.suggest_int("num_leaves", 20, 120),
            }
        )
    elif name == "catboost":
        params.update(
            {
                "iterations": trial.suggest_int("iterations", 150, 500),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            }
        )

    return params


# =====================
# METRICS
# =====================
def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


# =====================
# MAIN
# =====================
def main() -> None:
    args = parse_args()

    # ---- config (optional) ----
    if args.config and Path(args.config).exists():
        cfg = load_yaml(args.config)
    else:
        if args.config:
            logger.warning(f"Config not found at '{args.config}', using default config.")
        cfg = default_cfg()

    target = cfg.get("experiment", {}).get("target", "is_fraud")
    experiment_name = cfg.get("experiment", {}).get("name", "fraud-model-selection")
    resampling_cfg = cfg.get("resampling", {})
    use_smote = bool(resampling_cfg.get("use_smote", True))
    smote_random_state = int(resampling_cfg.get("random_state", 42))

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(f"{experiment_name}-tuning")

    # ---- best model name (optional) ----
    best_model_name = "logistic_regression"
    if Path(args.best_model_json).exists():
        info = load_json(args.best_model_json)
        best_model_name = info.get("best_model_name", best_model_name)
    else:
        logger.warning(f"best-model-json not found at '{args.best_model_json}', using {best_model_name}.")

    if best_model_name not in cfg["models"]:
        logger.warning(f"Model '{best_model_name}' not in cfg['models'], falling back to logistic_regression.")
        best_model_name = "logistic_regression"

    model_cfg = cfg["models"][best_model_name]
    dataset_branch = model_cfg.get("dataset", "log")
    base_params = model_cfg.get("params", {})

    # ---- fe_params (required) ----
    fe_params_path = Path(args.fe_params)
    if not fe_params_path.exists():
        raise FileNotFoundError(
            f"Missing required file: {args.fe_params}\n"
            f"Tip: run FeatureEngineering.py first to create models/trained/fe_params.pkl "
            f"or pass --fe-params with correct path."
        )
    fe_params = joblib.load(fe_params_path)

    # ---- load data ----
    train_log = load_table(args.train_log)
    test_log = load_table(args.test_log)
    train_tree = load_table(args.train_tree)
    test_tree = load_table(args.test_tree)

    if dataset_branch == "log":
        train, test = train_log, test_log
    else:
        train, test = train_tree, test_tree

    x_train, y_train = split_xy(train, target)
    x_test, y_test = split_xy(test, target)

    # ✅ FIX: drop datetime columns before SMOTE/training
    x_train = drop_datetime_columns(x_train)
    x_test = drop_datetime_columns(x_test)

    # SMOTE from config
    x_train, y_train = maybe_smote(x_train, y_train, use_smote, smote_random_state)

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, best_model_name, base_params)
        model = build_model(best_model_name, params)

        score = float(
            cross_val_score(
                model,
                x_train,
                y_train,
                cv=cv,
                scoring="average_precision",
            ).mean()
        )

        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            mlflow.log_param("selected_model", best_model_name)
            mlflow.log_param("dataset_branch", dataset_branch)
            mlflow.log_param("use_smote", use_smote)
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(params)
            mlflow.log_metric("cv_average_precision", score)

        return score

    with mlflow.start_run(run_name=f"tune_{best_model_name}"):
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("dataset_branch", dataset_branch)
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("cv", args.cv)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)

        best_params = suggest_params(
            optuna.trial.FixedTrial(study.best_params),
            best_model_name,
            base_params,
        )

        model = build_model(best_model_name, best_params)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_score)
        logger.info(f"Metrics: {metrics}")

        mlflow.log_metric("best_cv_average_precision", float(study.best_value))
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(metrics)

        # ---- save (match your screenshot) ----
        trained_dir = Path(args.models_dir) / "trained"
        artifact_dir = Path(args.models_dir) / "artifacts"
        trained_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_path = trained_dir / "fraud_model.pkl"
        joblib.dump(model, model_path)

        joblib.dump(fe_params, trained_dir / "fe_params.pkl")
        joblib.dump(x_train.columns.tolist(), trained_dir / "model_columns.pkl")

        # encoder/scaler artifacts (optional)
        if dataset_branch == "log":
            cat_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
            num_cols = [c for c in x_train.columns if c not in cat_cols]

            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            scaler = StandardScaler()

            if cat_cols:
                encoder.fit(x_train[cat_cols])
                joblib.dump(encoder, artifact_dir / "onehot_encoder.pkl")

            if num_cols:
                scaler.fit(x_train[num_cols])
                joblib.dump(scaler, artifact_dir / "scaler.pkl")
        else:
            # ✅ Fit label encoders based on known categorical columns, not dtype=object
            cate_cols = [
                "merchant_category",
                "merchant_country",
                "device_type",
                "mcc_code",
                "hour_of_day",
                "day_of_week",
            ]
            cate_cols = [c for c in cate_cols if c in x_train.columns]

            encoders: dict[str, LabelEncoder] = {}
            for col in cate_cols:
                le = LabelEncoder()
                le.fit(x_train[col].astype(str))
                encoders[col] = le

            joblib.dump(encoders, artifact_dir / "label_encoders.pkl")

        meta = {
            "selected_model": best_model_name,
            "dataset_branch": dataset_branch,
            "best_params": best_params,
            "best_cv_average_precision": float(study.best_value),
            "metrics": metrics,
            "model_path": str(model_path),
            "target": target,
            "use_smote": use_smote,
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }

        meta_path = trained_dir / "trained_model_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(meta_path), artifact_path="reports")
        mlflow.log_artifact(str(trained_dir / "fe_params.pkl"), artifact_path="reports")
        mlflow.log_artifact(str(trained_dir / "model_columns.pkl"), artifact_path="reports")

        if dataset_branch == "log":
            if (artifact_dir / "onehot_encoder.pkl").exists():
                mlflow.log_artifact(str(artifact_dir / "onehot_encoder.pkl"), artifact_path="reports")
            if (artifact_dir / "scaler.pkl").exists():
                mlflow.log_artifact(str(artifact_dir / "scaler.pkl"), artifact_path="reports")
        else:
            if (artifact_dir / "label_encoders.pkl").exists():
                mlflow.log_artifact(str(artifact_dir / "label_encoders.pkl"), artifact_path="reports")

        mlflow.sklearn.log_model(model, name="final_model")

        logger.info("Training complete. Artifacts saved.")
        logger.info(f"Saved model to: {model_path}")
        logger.info(f"MLflow run id: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()