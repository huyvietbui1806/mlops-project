from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline models and select best model family.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train-log", type=str, required=True)
    parser.add_argument("--test-log", type=str, required=True)
    parser.add_argument("--train-tree", type=str, required=True)
    parser.add_argument("--test-tree", type=str, required=True)
    parser.add_argument("--reports-dir", type=str, default="reports/training")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file format: {p.suffix}")

def drop_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm này loại bỏ tất cả các cột kiểu datetime trong dataframe.
    """
    df = df.copy()
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if dt_cols:
        logger.warning(f"Dropping datetime columns before SMOTE/training: {dt_cols}")
        df = df.drop(columns=dt_cols)

    if "timestamp" in df.columns and df["timestamp"].dtype == "object":
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed.notna().any():
            logger.warning("Dropping 'timestamp' column (parsed as datetime) before SMOTE/training.")
            df = df.drop(columns=["timestamp"])

    return df

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


def get_model_instance(model_name: str, params: dict[str, Any]) -> Any:
    model_map = {
        "logistic_regression": LogisticRegression,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
        "catboost": CatBoostClassifier,
    }
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}")
    return model_map[model_name](**params)


def build_stacking_model(random_state: int = 42) -> StackingClassifier:
    estimators = [
        (
            "xgb",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                objective="binary",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "cat",
            CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_state,
                verbose=0,
            ),
        ),
    ]

    final_estimator = LogisticRegression(
        max_iter=300,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
    )

    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method="predict_proba",
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def build_feature_importance_df(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        return (
            pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_).ravel()
        return (
            pd.DataFrame({"feature": feature_names, "importance": coef})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return None


def log_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, artifact_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False, cmap=plt.cm.Blues)
    ax.set_title(title)
    plt.tight_layout()
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)


def log_pr_curve(y_true: pd.Series, y_score: np.ndarray, artifact_path: str, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)


def log_roc_curve(y_true: pd.Series, y_score: np.ndarray, artifact_path: str, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)


def log_feature_importance(importance_df: pd.DataFrame, artifact_path: str, title: str, top_n: int = 15) -> None:
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)


def train_and_log(
    model_name: str,
    display_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    selection_metric: str,
    use_smote: bool,
    smote_random_state: int,
) -> dict[str, Any]:
    x_fit, y_fit = maybe_apply_smote(x_train, y_train, use_smote, smote_random_state)

    with mlflow.start_run(run_name=model_name):
        model.fit(x_fit, y_fit)

        y_pred = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 1]

        y_pred = np.array(y_pred).reshape(-1).astype(int)
        y_score = np.array(y_score).reshape(-1)

        metrics = compute_metrics(y_test, y_pred, y_score)

        if hasattr(model, "get_params"):
            try:
                mlflow.log_params(model.get_params())
            except Exception:
                pass

        mlflow.log_param("use_smote", use_smote)
        mlflow.log_metrics(metrics)

        log_confusion_matrix(
            y_test, y_pred,
            artifact_path=f"plots/{model_name}/confusion_matrix.png",
            title=f"Confusion Matrix - {display_name}",
        )
        log_pr_curve(
            y_test, y_score,
            artifact_path=f"plots/{model_name}/pr_curve.png",
            title=f"PR Curve - {display_name}",
        )
        log_roc_curve(
            y_test, y_score,
            artifact_path=f"plots/{model_name}/roc_curve.png",
            title=f"ROC Curve - {display_name}",
        )

        importance_df = build_feature_importance_df(model, x_train.columns.tolist())
        if importance_df is not None:
            csv_path = Path("temp_feature_importance.csv")
            importance_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path=f"reports/{model_name}")
            log_feature_importance(
                importance_df,
                artifact_path=f"plots/{model_name}/feature_importance.png",
                title=f"Feature Importance - {display_name}",
            )
            if csv_path.exists():
                csv_path.unlink()

        return {
            "model_name": model_name,
            "display_name": display_name,
            "selection_score": metrics[selection_metric],
            **metrics,
            "use_smote": use_smote,
            "run_id": mlflow.active_run().info.run_id,
        }


def main(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)

    experiment_cfg = cfg["experiment"]
    target = experiment_cfg["target"]
    experiment_name = experiment_cfg["name"]
    selection_metric = experiment_cfg["selection_metric"]
    random_state = experiment_cfg.get("random_state", 42)

    resampling_cfg = cfg.get("resampling", {})
    use_smote = bool(resampling_cfg.get("use_smote", False))
    smote_random_state = int(resampling_cfg.get("random_state", 42))

    if args.mlflow_tracking_uri and args.mlflow_tracking_uri.strip():
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_log = load_table(args.train_log)
    test_log = load_table(args.test_log)
    train_tree = load_table(args.train_tree)
    test_tree = load_table(args.test_tree)

    x_train_log, y_train_log = split_xy(train_log, target)
    x_test_log, y_test_log = split_xy(test_log, target)

    # Loại bỏ cột datetime trước khi huấn luyện
    x_train_log = drop_datetime_columns(x_train_log)
    x_test_log = drop_datetime_columns(x_test_log)

    x_train_tree, y_train_tree = split_xy(train_tree, target)
    x_test_tree, y_test_tree = split_xy(test_tree, target)

    # Loại bỏ cột datetime trong cây
    x_train_tree = drop_datetime_columns(x_train_tree)
    x_test_tree = drop_datetime_columns(x_test_tree)

    results: list[dict[str, Any]] = []

    for model_name, model_cfg in cfg["models"].items():
        if not model_cfg.get("enabled", False):
            continue

        dataset_branch = model_cfg["dataset"]
        params = model_cfg["params"]
        model = get_model_instance(model_name, params)

        if dataset_branch == "log":
            x_train, y_train, x_test, y_test = x_train_log, y_train_log, x_test_log, y_test_log
        elif dataset_branch == "tree":
            x_train, y_train, x_test, y_test = x_train_tree, y_train_tree, x_test_tree, y_test_tree
        else:
            raise ValueError(f"Unsupported dataset branch: {dataset_branch}")

        display_name = model_name.replace("_", " ").title()
        result = train_and_log(
            model_name=model_name,
            display_name=display_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            selection_metric=selection_metric,
            use_smote=use_smote,
            smote_random_state=smote_random_state,
        )
        result["dataset"] = dataset_branch
        result["baseline_params"] = params
        results.append(result)

    if cfg.get("stacking", {}).get("enabled", False):
        stacking_dataset = cfg["stacking"].get("dataset", "tree")
        if stacking_dataset != "tree":
            raise ValueError("Stacking currently supports only tree dataset branch.")

        stacking_model = build_stacking_model(random_state=random_state)
        result = train_and_log(
            model_name="stacking",
            display_name="Stacking Ensemble",
            model=stacking_model,
            x_train=x_train_tree,
            y_train=y_train_tree,
            x_test=x_test_tree,
            y_test=y_test_tree,
            selection_metric=selection_metric,
            use_smote=use_smote,
            smote_random_state=smote_random_state,
        )
        result["dataset"] = "tree"
        result["baseline_params"] = {"type": "stacking"}
        results.append(result)

    leaderboard = pd.DataFrame(results).sort_values(
        by="selection_score",
        ascending=False,
    ).reset_index(drop=True)

    logger.info("\n=== Leaderboard ===")
    logger.info(
        "\n%s",
        leaderboard[["model_name", "dataset", "use_smote", "pr_auc", "roc_auc", "f1_score", "precision", "recall"]],
    )

    best = leaderboard.iloc[0].to_dict()

    best_model_info = {
    "best_model_name": best["model_name"],
    "dataset": best["dataset"],
    "use_smote": bool(best["use_smote"]),
    "selection_score": float(best["selection_score"]),
    "pr_auc": float(best["pr_auc"]),
    "roc_auc": float(best["roc_auc"]),
    "f1_score": float(best["f1_score"]),
    "precision": float(best["precision"]),
    "recall": float(best["recall"]),
    "run_id": best["run_id"],
}

    best_model_json_path = reports_dir / "best_model.json"
    with open(best_model_json_path, "w", encoding="utf-8") as f:
        json.dump(best_model_info, f, ensure_ascii=False, indent=2)

    logger.info("Saved best model info to: %s", best_model_json_path)

    logger.info("\n=== Best Model Family ===")
    logger.info("model_name      : %s", best["model_name"])
    logger.info("dataset         : %s", best["dataset"])
    logger.info("use_smote       : %s", best["use_smote"])
    logger.info("%s : %.6f", selection_metric, best["selection_score"])
    logger.info("run_id          : %s", best["run_id"])

    leaderboard_path = reports_dir / "baseline_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    logger.info("Saved leaderboard to: %s", leaderboard_path)

    print("\nCOPY THIS FOR TUNING:")
    print(
        "python src/models/tune_model.py `\n"
        f"  --config {args.config} `\n"
        f"  --best-model-json {reports_dir / 'best_model.json'} `\n"
        f"  --train-log {args.train_log} `\n"
        f"  --test-log {args.test_log} `\n"
        f"  --train-tree {args.train_tree} `\n"
        f"  --test-tree {args.test_tree} `\n"
        "  --models-dir models `\n"
        f"  --mlflow-tracking-uri {args.mlflow_tracking_uri or 'http://localhost:5555'} `\n"
        "  --n-trials 10 `\n"
        "  --cv 3"
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)