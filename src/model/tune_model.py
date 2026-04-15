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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune selected model family with Optuna.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument(
        "--best-model-json",
        type=str,
        required=True,
        help="Path to JSON file containing best_model_name",
    )
    parser.add_argument("--train-log", type=str, required=True)
    parser.add_argument("--test-log", type=str, required=True)
    parser.add_argument("--train-tree", type=str, required=True)
    parser.add_argument("--test-tree", type=str, required=True)
    parser.add_argument("--models-dir", type=str, required=True)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--cv", type=int, default=3)
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)

    raise ValueError(f"Unsupported file format: {p.suffix}")


def split_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")
    x = df.drop(columns=[target])
    y = df[target].astype(int)
    return x, y


def maybe_apply_smote(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if not use_smote:
        return x_train, y_train

    smote = SMOTE(random_state=random_state)
    x_res, y_res = smote.fit_resample(x_train, y_train)
    x_res = pd.DataFrame(x_res, columns=x_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    return x_res, y_res


def build_model(model_name: str, params: dict[str, Any]) -> Any:
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    if model_name == "xgboost":
        return XGBClassifier(**params)
    if model_name == "lightgbm":
        return LGBMClassifier(**params)
    if model_name == "catboost":
        return CatBoostClassifier(**params)

    raise ValueError(f"Unsupported model: {model_name}")


def suggest_params(
    trial: optuna.Trial,
    model_name: str,
    baseline_params: dict[str, Any],
) -> dict[str, Any]:
    params = dict(baseline_params)

    if model_name == "logistic_regression":
        params.update(
            {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 1000),
            }
        )
        return params

    if model_name == "xgboost":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }
        )
        return params

    if model_name == "lightgbm":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500),
                "num_leaves": trial.suggest_int("num_leaves", 20, 120),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            }
        )
        return params

    if model_name == "catboost":
        params.update(
            {
                "iterations": trial.suggest_int("iterations", 150, 500),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }
        )
        return params

    raise ValueError(f"No tuning space defined for model: {model_name}")


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def main(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    best_model_info = load_json(args.best_model_json)

    experiment_cfg = cfg["experiment"]
    target = experiment_cfg["target"]
    experiment_name = experiment_cfg["name"]
    random_state = int(experiment_cfg.get("random_state", 42))

    resampling_cfg = cfg.get("resampling", {})
    use_smote = bool(resampling_cfg.get("use_smote", False))
    smote_random_state = int(resampling_cfg.get("random_state", 42))

    model_name = best_model_info["best_model_name"]
    if model_name not in cfg["models"]:
        raise ValueError(f"Model '{model_name}' not found in config file")

    model_cfg = cfg["models"][model_name]
    dataset_branch = model_cfg["dataset"]
    baseline_params = model_cfg["params"]

    if args.mlflow_tracking_uri and args.mlflow_tracking_uri.strip():
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.set_experiment(f"{experiment_name}-tuning")

    train_log = load_table(args.train_log)
    test_log = load_table(args.test_log)
    train_tree = load_table(args.train_tree)
    test_tree = load_table(args.test_tree)

    x_train_log, y_train_log = split_xy(train_log, target)
    x_test_log, y_test_log = split_xy(test_log, target)

    x_train_tree, y_train_tree = split_xy(train_tree, target)
    x_test_tree, y_test_tree = split_xy(test_tree, target)

    if dataset_branch == "log":
        x_train, y_train, x_test, y_test = x_train_log, y_train_log, x_test_log, y_test_log
    elif dataset_branch == "tree":
        x_train, y_train, x_test, y_test = x_train_tree, y_train_tree, x_test_tree, y_test_tree
    else:
        raise ValueError(f"Unsupported dataset branch: {dataset_branch}")

    x_fit, y_fit = maybe_apply_smote(x_train, y_train, use_smote, smote_random_state)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name, baseline_params)
        model = build_model(model_name, params)

        scores = cross_val_score(
            model,
            x_fit,
            y_fit,
            cv=cv,
            scoring="average_precision",
            n_jobs=1,
        )
        mean_score = float(np.mean(scores))

        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            mlflow.log_param("selected_model", model_name)
            mlflow.log_param("dataset_branch", dataset_branch)
            mlflow.log_param("use_smote", use_smote)
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(params)
            mlflow.log_metric("cv_average_precision", mean_score)

        return mean_score

    with mlflow.start_run(run_name=f"tune_{model_name}"):
        mlflow.log_param("selected_model", model_name)
        mlflow.log_param("dataset_branch", dataset_branch)
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("cv", args.cv)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)

        best_params = suggest_params(
            trial=optuna.trial.FixedTrial(study.best_params),
            model_name=model_name,
            baseline_params=baseline_params,
        )

        best_model = build_model(model_name, best_params)
        best_model.fit(x_fit, y_fit)

        y_pred = best_model.predict(x_test)
        y_score = best_model.predict_proba(x_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_score)

        mlflow.log_metric("best_cv_average_precision", float(study.best_value))
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(metrics)

        models_dir = Path(args.models_dir) / "trained"
        models_dir.mkdir(parents=True, exist_ok=True)

        final_model_path = models_dir / "trained_model.pkl"
        joblib.dump(best_model, final_model_path)

        meta = {
            "selected_model": model_name,
            "dataset_branch": dataset_branch,
            "use_smote": use_smote,
            "n_trials": args.n_trials,
            "best_params": best_params,
            "best_cv_average_precision": float(study.best_value),
            "test_metrics": metrics,
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }

        meta_path = models_dir / "trained_model_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(str(final_model_path), artifact_path="models")
        mlflow.log_artifact(str(meta_path), artifact_path="reports")

        logger.info("Selected model : %s", model_name)
        logger.info("Best params    : %s", best_params)
        logger.info("Best CV score  : %.6f", study.best_value)
        logger.info("Test PR-AUC    : %.6f", metrics["pr_auc"])
        logger.info("Saved model to : %s", final_model_path)
        logger.info("Loaded best model name from JSON: %s", args.best_model_json)


if __name__ == "__main__":
    args = parse_args()
    main(args)