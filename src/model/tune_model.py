from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
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
    precision_score,
    recall_score,
    roc_auc_score,
)
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
    parser = argparse.ArgumentParser(
        description="Tune best model using train/valid and evaluate final metrics on test."
    )

    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--best-model-json", default="reports/training/best_model.json")

    parser.add_argument("--train-log", default="data/processed/train_log.parquet")
    parser.add_argument("--valid-log", default="data/processed/valid_log.parquet")
    parser.add_argument("--test-log", default="data/processed/test_log.parquet")

    parser.add_argument("--train-tree", default="data/processed/train_tree.parquet")
    parser.add_argument("--valid-tree", default="data/processed/valid_tree.parquet")
    parser.add_argument("--test-tree", default="data/processed/test_tree.parquet")

    parser.add_argument("--fe-params", default="models/trained/fe_params.pkl")
    parser.add_argument("--models-dir", default="models")

    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5555")
    parser.add_argument("--n-trials", type=int, default=2)

    parser.add_argument("--precision-floor", type=float, default=0.65)
    parser.add_argument("--threshold-step", type=float, default=0.01)

    return parser.parse_args()


# =====================
# UTILS
# =====================
def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}\n"
            f"Tip: run FeatureEngineering.py first or pass correct paths via CLI."
        )

    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
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
    return pd.DataFrame(x_res, columns=x.columns), pd.Series(y_res, name=y.name)


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
    df = df.copy()

    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if dt_cols:
        logger.warning("Dropping datetime columns before SMOTE/training: %s", dt_cols)
        df = df.drop(columns=dt_cols)

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
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def compute_threshold_metrics(y_true, y_score, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def find_best_threshold(
    y_true,
    y_score,
    precision_floor: float = 0.65,
    step: float = 0.01,
) -> dict[str, float | bool]:
    best: dict[str, float | bool] | None = None

    thresholds = np.arange(0.0, 1.0 + (step / 2), step)

    for threshold in thresholds:
        threshold = float(round(float(threshold), 4))
        y_pred = (y_score >= threshold).astype(int)

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        accuracy = float(accuracy_score(y_true, y_pred))

        if precision < precision_floor:
            continue

        candidate = {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "found": True,
        }

        if best is None:
            best = candidate
            continue

        if candidate["recall"] > best["recall"]:
            best = candidate
        elif candidate["recall"] == best["recall"] and candidate["precision"] > best["precision"]:
            best = candidate
        elif (
            candidate["recall"] == best["recall"]
            and candidate["precision"] == best["precision"]
            and candidate["f1"] > best["f1"]
        ):
            best = candidate
        elif (
            candidate["recall"] == best["recall"]
            and candidate["precision"] == best["precision"]
            and candidate["f1"] == best["f1"]
            and candidate["threshold"] < best["threshold"]
        ):
            best = candidate

    if best is not None:
        return best

    fallback_threshold = 0.5
    fallback_pred = (y_score >= fallback_threshold).astype(int)
    return {
        "threshold": float(fallback_threshold),
        "accuracy": float(accuracy_score(y_true, fallback_pred)),
        "precision": float(precision_score(y_true, fallback_pred, zero_division=0)),
        "recall": float(recall_score(y_true, fallback_pred, zero_division=0)),
        "f1": float(f1_score(y_true, fallback_pred, zero_division=0)),
        "found": False,
    }


# =====================
# MAIN
# =====================
def main() -> None:
    args = parse_args()

    if args.config and Path(args.config).exists():
        cfg = load_yaml(args.config)
    else:
        if args.config:
            logger.warning("Config not found at '%s', using default config.", args.config)
        cfg = default_cfg()

    target = cfg.get("experiment", {}).get("target", "is_fraud")
    experiment_name = cfg.get("experiment", {}).get("name", "fraud-model-selection")
    resampling_cfg = cfg.get("resampling", {})
    use_smote = bool(resampling_cfg.get("use_smote", True))
    smote_random_state = int(resampling_cfg.get("random_state", 42))

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(f"{experiment_name}-tuning")

    best_model_name = "logistic_regression"
    if Path(args.best_model_json).exists():
        info = load_json(args.best_model_json)
        best_model_name = info.get("best_model_name", best_model_name)
    else:
        logger.warning("best-model-json not found at '%s', using %s.", args.best_model_json, best_model_name)

    if best_model_name not in cfg["models"]:
        logger.warning("Model '%s' not in cfg['models'], fallback to logistic_regression.", best_model_name)
        best_model_name = "logistic_regression"

    model_cfg = cfg["models"][best_model_name]
    dataset_branch = model_cfg.get("dataset", "log")
    base_params = model_cfg.get("params", {})

    fe_params_path = Path(args.fe_params)
    if not fe_params_path.exists():
        raise FileNotFoundError(
            f"Missing required file: {args.fe_params}\n"
            f"Tip: run FeatureEngineering.py first to create models/trained/fe_params.pkl."
        )
    fe_params = joblib.load(fe_params_path)

    train_log = load_table(args.train_log)
    valid_log = load_table(args.valid_log)
    test_log = load_table(args.test_log)

    train_tree = load_table(args.train_tree)
    valid_tree = load_table(args.valid_tree)
    test_tree = load_table(args.test_tree)

    if dataset_branch == "log":
        train, valid, test = train_log, valid_log, test_log
    else:
        train, valid, test = train_tree, valid_tree, test_tree

    x_train, y_train = split_xy(train, target)
    x_valid, y_valid = split_xy(valid, target)
    x_test, y_test = split_xy(test, target)

    x_train = drop_datetime_columns(x_train)
    x_valid = drop_datetime_columns(x_valid)
    x_test = drop_datetime_columns(x_test)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, best_model_name, base_params)
        model = build_model(best_model_name, params)

        x_fit, y_fit = maybe_smote(x_train, y_train, use_smote, smote_random_state)
        model.fit(x_fit, y_fit)

        y_valid_score = model.predict_proba(x_valid)[:, 1]
        y_valid_pred = model.predict(x_valid)

        metrics = compute_metrics(y_valid, y_valid_pred, y_valid_score)
        score = float(metrics["pr_auc"])

        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            mlflow.log_param("selected_model", best_model_name)
            mlflow.log_param("dataset_branch", dataset_branch)
            mlflow.log_param("use_smote", use_smote)
            mlflow.log_param("fit_split", "train")
            mlflow.log_param("tuning_split", "valid")
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(params)
            mlflow.log_metrics({f"valid_{k}": v for k, v in metrics.items()})

        return score

    with mlflow.start_run(run_name=f"tune_{best_model_name}"):
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("dataset_branch", dataset_branch)
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_param("fit_split", "train")
        mlflow.log_param("tuning_split", "valid")
        mlflow.log_param("final_metrics_split", "test")
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("precision_floor", args.precision_floor)
        mlflow.log_param("threshold_step", args.threshold_step)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)

        best_params = suggest_params(
            optuna.trial.FixedTrial(study.best_params),
            best_model_name,
            base_params,
        )

        model = build_model(best_model_name, best_params)

        x_fit, y_fit = maybe_smote(x_train, y_train, use_smote, smote_random_state)
        model.fit(x_fit, y_fit)

        # ----- VALID metrics for threshold tuning -----
        y_valid_pred = model.predict(x_valid)
        y_valid_score = model.predict_proba(x_valid)[:, 1]

        valid_default_metrics = compute_metrics(y_valid, y_valid_pred, y_valid_score)
        logger.info("Validation metrics (default predict on valid): %s", valid_default_metrics)

        threshold_result = find_best_threshold(
            y_true=y_valid,
            y_score=y_valid_score,
            precision_floor=args.precision_floor,
            step=args.threshold_step,
        )
        selected_threshold = float(threshold_result["threshold"])

        valid_threshold_metrics = compute_threshold_metrics(
            y_true=y_valid,
            y_score=y_valid_score,
            threshold=selected_threshold,
        )
        logger.info("Validation metrics (threshold-selected on valid): %s", valid_threshold_metrics)

        # ----- TEST metrics final -----
        y_test_pred = model.predict(x_test)
        y_test_score = model.predict_proba(x_test)[:, 1]

        test_default_metrics = compute_metrics(y_test, y_test_pred, y_test_score)
        test_threshold_metrics = compute_threshold_metrics(
            y_true=y_test,
            y_score=y_test_score,
            threshold=selected_threshold,
        )

        logger.info("Test metrics (default predict on test): %s", test_default_metrics)
        logger.info("Test metrics (threshold-selected from valid): %s", test_threshold_metrics)

        mlflow.log_metric("best_valid_pr_auc", float(study.best_value))
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        mlflow.log_metrics({f"valid_default_{k}": v for k, v in valid_default_metrics.items()})
        mlflow.log_metrics({f"valid_threshold_{k}": v for k, v in valid_threshold_metrics.items() if k != "threshold"})
        mlflow.log_metrics({f"test_default_{k}": v for k, v in test_default_metrics.items()})
        mlflow.log_metrics({f"test_threshold_{k}": v for k, v in test_threshold_metrics.items() if k != "threshold"})

        mlflow.log_metric("selected_threshold", selected_threshold)
        mlflow.log_metric(
            "threshold_found_under_precision_floor",
            float(bool(threshold_result["found"])),
        )

        trained_dir = Path(args.models_dir) / "trained"
        trained_dir.mkdir(parents=True, exist_ok=True)

        model_path = trained_dir / "trained_model.pkl"
        joblib.dump(model, model_path)

        joblib.dump(fe_params, trained_dir / "fe_params.pkl")
        joblib.dump(x_train.columns.tolist(), trained_dir / "model_columns.pkl")

        meta = {
            "selected_model": best_model_name,
            "dataset_branch": dataset_branch,
            "best_params": best_params,
            "best_valid_pr_auc": float(study.best_value),
            "threshold": selected_threshold,
            "threshold_search": {
                "precision_floor": float(args.precision_floor),
                "step": float(args.threshold_step),
                "found_threshold_meeting_precision_floor": bool(threshold_result["found"]),
            },
            "metrics_default_threshold_on_valid": valid_default_metrics,
            "metrics_threshold_selected_on_valid": valid_threshold_metrics,
            "metrics_default_threshold_on_test": test_default_metrics,
            "metrics_threshold_selected_on_test": test_threshold_metrics,
            "model_path": str(model_path),
            "target": target,
            "use_smote": use_smote,
            "fit_split": "train",
            "tuning_split": "valid",
            "final_metrics_split": "test",
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }

        meta_path = trained_dir / "trained_model_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(meta_path), artifact_path="reports")
        mlflow.log_artifact(str(trained_dir / "fe_params.pkl"), artifact_path="reports")
        mlflow.log_artifact(str(trained_dir / "model_columns.pkl"), artifact_path="reports")

        mlflow.sklearn.log_model(model, name="final_model")

        logger.info("Training complete. Artifacts saved.")
        logger.info("Saved model to: %s", model_path)
        logger.info("Saved meta to: %s", meta_path)
        logger.info("MLflow run id: %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()