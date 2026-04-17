from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..features.FeatureEngineering import add_features
from .schemas import FraudDetectionRequest, FraudResponse

# =====================
# PATHS — khớp tune_model.py (trained + artifacts tách riêng)
# =====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_DIR = MODELS_DIR / "trained"
ARTIFACT_DIR = MODELS_DIR / "artifacts"

BEST_MODEL_JSON = PROJECT_ROOT / "reports" / "training" / "best_model.json"
META_PATH = TRAINED_DIR / "trained_model_meta.json"
MODEL_PATH = TRAINED_DIR / "trained_model.pkl"
FE_PARAMS_PATH = TRAINED_DIR / "fe_params.pkl"
COLUMNS_PATH = TRAINED_DIR / "model_columns.pkl"

THRESHOLD = 0.65

# Các field identifier — không được đưa vào model
_ID_COLS = {"transaction_id", "user_id"}


# =====================
# LOAD HELPERS
# =====================
def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_artifacts() -> dict:
    """Load toàn bộ artifact. Raise RuntimeError nếu thiếu file."""
    required = [META_PATH, MODEL_PATH, FE_PARAMS_PATH, COLUMNS_PATH]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Artifact files not found. Have you run FeatureEngineering.py and tune_model.py?\n"
            f"Missing: {[str(p) for p in missing]}"
        )

    meta = _load_json(META_PATH)
    model = joblib.load(MODEL_PATH)
    fe_params = joblib.load(FE_PARAMS_PATH)
    model_columns = joblib.load(COLUMNS_PATH)

    artifacts: dict[str, object] = {
        "meta": meta,
        "model": model,
        "fe_params": fe_params,
        "model_columns": model_columns,
    }

    branch = meta.get("dataset_branch")

    if branch == "log":
        enc_path = ARTIFACT_DIR / "onehot_encoder.pkl"
        scaler_path = ARTIFACT_DIR / "scaler.pkl"
        missing_art = [p for p in [enc_path, scaler_path] if not p.exists()]
        if missing_art:
            raise RuntimeError(f"Log-branch artifacts missing: {[str(p) for p in missing_art]}")
        artifacts["encoder"] = joblib.load(enc_path)
        artifacts["scaler"] = joblib.load(scaler_path)

    elif branch == "tree":
        le_path = ARTIFACT_DIR / "label_encoders.pkl"
        if not le_path.exists():
            raise RuntimeError(
                "Tree-branch artifact missing: label_encoders.pkl\n"
                f"Expected at: {le_path}\n"
                "Fix: re-run tune_model.py with a tree model (xgboost/lightgbm/catboost) so it saves label_encoders.pkl."
            )
        artifacts["label_encoders"] = joblib.load(le_path)

    else:
        raise RuntimeError(f"Unknown dataset_branch in meta: '{branch}'")

    return artifacts


# =====================
# GLOBAL LOAD
# =====================
ARTIFACTS: dict = load_artifacts()

_meta = ARTIFACTS["meta"]
_model = ARTIFACTS["model"]
_fe_params = ARTIFACTS["fe_params"]
_model_columns = ARTIFACTS["model_columns"]


# =====================
# BUILD INPUT
# =====================
def _build_raw_df(request: FraudDetectionRequest) -> pd.DataFrame:
    data = request.dict()
    for col in _ID_COLS:
        data.pop(col, None)
    return pd.DataFrame([data])


def _build_raw_df_batch(requests: list[FraudDetectionRequest]) -> pd.DataFrame:
    rows = []
    for req in requests:
        data = req.dict()
        for col in _ID_COLS:
            data.pop(col, None)
        rows.append(data)
    return pd.DataFrame(rows)


def _drop_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror tune_model.py behavior:
    - drop datetime64 cols
    - if timestamp parses as datetime, drop it too
    """
    df = df.copy()

    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if dt_cols:
        df = df.drop(columns=dt_cols)

    if "timestamp" in df.columns and df["timestamp"].dtype == "object":
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed.notna().any():
            df = df.drop(columns=["timestamp"])

    return df


# =====================
# BRANCH PREPROCESS
# =====================
def _preprocess_log(df: pd.DataFrame) -> pd.DataFrame:
    """OneHot encode + StandardScaler cho logistic branch."""
    encoder = ARTIFACTS["encoder"]
    scaler = ARTIFACTS["scaler"]

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    if cat_cols:
        encoded_arr = encoder.transform(df[cat_cols])
        feature_names = encoder.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded_arr, columns=feature_names, index=df.index)
    else:
        encoded_df = pd.DataFrame(index=df.index)

    if num_cols:
        scaled_arr = scaler.transform(df[num_cols])
        scaled_df = pd.DataFrame(scaled_arr, columns=num_cols, index=df.index)
    else:
        scaled_df = pd.DataFrame(index=df.index)

    return pd.concat([scaled_df, encoded_df], axis=1)


def _preprocess_tree(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode cho tree branch; unknown -> 'UNKNOWN'."""
    df = df.copy()
    encoders = ARTIFACTS["label_encoders"]

    for col, le in encoders.items():
        if col not in df.columns:
            df[col] = 0
            continue

        df[col] = df[col].astype(str)

        # ensure UNKNOWN exists
        known = set(le.classes_)
        if "UNKNOWN" not in known:
            le.classes_ = np.append(le.classes_, "UNKNOWN")
            known = set(le.classes_)

        df[col] = df[col].map(lambda x: x if x in known else "UNKNOWN")
        df[col] = le.transform(df[col])

    return df


def _preprocess_by_branch(df: pd.DataFrame) -> pd.DataFrame:
    branch = _meta.get("dataset_branch")
    if branch == "log":
        return _preprocess_log(df)
    if branch == "tree":
        return _preprocess_tree(df)
    raise ValueError(f"Unknown dataset_branch: '{branch}'")


# =====================
# BUSINESS RULES
# =====================
def _apply_business_rules(df: pd.DataFrame) -> list[str]:
    """
    Input df: after add_features, before preprocess/reindex.
    """
    row = df.iloc[0]
    rules: list[str] = []

    rule_map = {
        "high_amount_flag": "high_amount",
        "is_night": "night_transaction",
        "high_ip_risk_flag": "high_ip_risk",
        "high_velocity_1h_flag": "velocity_spike",
        "high_utilization_flag": "high_utilization",
    }

    for col, label in rule_map.items():
        if col in df.columns and row[col] == 1:
            rules.append(label)

    return rules


def _get_risk_level(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


# =====================
# PREDICT — single
# =====================
def predict_fraud(request: FraudDetectionRequest) -> FraudResponse:
    df = _build_raw_df(request)

    # Feature engineering
    df, _ = add_features(df, _fe_params)

    # mirror tune_model: remove datetime features (if any)
    df = _drop_datetime_columns(df)

    # business rules before preprocess
    rules_triggered = _apply_business_rules(df)

    # preprocess
    df = _preprocess_by_branch(df)

    # align columns exactly like training
    df = df.reindex(columns=_model_columns, fill_value=0)

    fraud_score = float(_model.predict_proba(df)[0][1])

    return FraudResponse(
        is_fraud=fraud_score >= THRESHOLD,
        fraud_score=round(fraud_score, 4),
        risk_level=_get_risk_level(fraud_score),
        triggered_rules=rules_triggered,
        prediction_time=datetime.now(),
    )


# =====================
# PREDICT — batch
# =====================
def batch_predict(requests: list[FraudDetectionRequest]) -> list[FraudResponse]:
    if not requests:
        return []

    df = _build_raw_df_batch(requests)

    df, _ = add_features(df, _fe_params)

    df = _drop_datetime_columns(df)

    rules_per_row = [_apply_business_rules(df.iloc[[i]]) for i in range(len(df))]

    df = _preprocess_by_branch(df)
    df = df.reindex(columns=_model_columns, fill_value=0)

    scores = _model.predict_proba(df)[:, 1]

    return [
        FraudResponse(
            is_fraud=float(score) >= THRESHOLD,
            fraud_score=round(float(score), 4),
            risk_level=_get_risk_level(float(score)),
            triggered_rules=rules,
            prediction_time=datetime.now(),
        )
        for score, rules in zip(scores, rules_per_row)
    ]