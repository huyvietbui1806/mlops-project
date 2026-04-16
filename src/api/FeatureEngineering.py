from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


TARGET = "is_fraud"

NUM_COLS = [
    "account_age_days",
    "time_since_last_s",
    "log_amount",
    "is_round_amount",
    "high_amount_flag",
    "log_ip_risk_score",
    "high_ip_risk_flag",
    "log_amount_vs_avg_ratio",
    "high_amount_vs_avg_flag",
    "log_velocity_1h",
    "high_velocity_1h_flag",
    "utilization",
    "high_utilization_flag",
]

CATE_COLS = [
    "merchant_category",
    "merchant_country",
    "device_type",
    "mcc_code",
    "hour_of_day",
    "day_of_week",
]


def parse_args() -> argparse.Namespace:
    """
    Defaults are set so you can run:
      python FeatureEngineering.py

    You can still override everything via CLI flags.
    """
    parser = argparse.ArgumentParser(description="Feature engineering for fraud detection")

    # Inputs (defaults to your repo layout shown in screenshot)
    parser.add_argument("--train-input", type=str, default="data/csv/train.csv")
    parser.add_argument("--test-input", type=str, default="data/csv/test.csv")

    # Outputs
    parser.add_argument("--train-log-output", type=str, default="data/processed/train_log.parquet")
    parser.add_argument("--test-log-output", type=str, default="data/processed/test_log.parquet")
    parser.add_argument("--train-tree-output", type=str, default="data/processed/train_tree.parquet")
    parser.add_argument("--test-tree-output", type=str, default="data/processed/test_tree.parquet")

    # FeatureEngineering params artifact (match your screenshot convention)
    parser.add_argument("--fe-params-output", type=str, default="models/trained/fe_params.pkl")

    return parser.parse_args()


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_table(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def safe_divide(num: pd.Series, denom: pd.Series, fill: float = 0) -> pd.Series:
    return (num / denom.replace(0, 1e-10)).replace([np.inf, -np.inf], fill).fillna(fill)


def safe_log(s: pd.Series, fill: float = 0) -> pd.Series:
    return pd.Series(np.log1p(np.maximum(s, 0)), index=s.index).replace([np.inf, -np.inf], fill).fillna(fill)


def basic_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=["fraud_pattern", "transaction_id", "account_id"], errors="ignore")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(by="timestamp")
        # Keep timestamp as a normal column for parquet saving (index=False below),
        # but still ensure consistent ordering.
        # If you truly want it as index, uncomment:
        # df = df.set_index("timestamp")
    else:
        # If timestamp is missing, keep as-is
        pass

    return df


def fit_params(df: pd.DataFrame) -> dict:
    params: dict = {}

    if "amount" in df.columns:
        params["amount_q95"] = df["amount"].quantile(0.95)
    else:
        params["amount_q95"] = 0

    if "ip_risk_score" in df.columns:
        params["ip_q80"] = df["ip_risk_score"].quantile(0.80)

    if "amount_vs_avg_ratio" in df.columns:
        params["ratio_q90"] = df["amount_vs_avg_ratio"].quantile(0.90)

    if "velocity_1h" in df.columns:
        params["velocity_q90"] = df["velocity_1h"].quantile(0.90)

    if "credit_limit" in df.columns and "amount" in df.columns:
        util = df["amount"] / df["credit_limit"].replace(0, np.nan)
        params["util_q90"] = util.quantile(0.90)

    return params


def add_features(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    new_var: list[str] = []

    # 1. Amount features
    if "amount" in df.columns:
        df["log_amount"] = safe_log(df["amount"])
        df["is_round_amount"] = ((df["amount"] % 10 == 0) & (df["amount"] > 0)).astype(int)
        df["high_amount_flag"] = (df["amount"] > params.get("amount_q95", 0)).astype(int)
        new_var += ["log_amount", "is_round_amount", "high_amount_flag"]

    # 2. Time features
    if "hour_of_day" in df.columns:
        df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
        df["is_business_hours"] = df["hour_of_day"].between(9, 17).astype(int)
        new_var += ["is_night", "is_business_hours"]

    # 3. Risk features
    if "ip_risk_score" in df.columns and "ip_q80" in params:
        df["log_ip_risk_score"] = safe_log(df["ip_risk_score"])
        df["high_ip_risk_flag"] = (df["ip_risk_score"] > params["ip_q80"]).astype(int)
        new_var += ["log_ip_risk_score", "high_ip_risk_flag"]
        df.drop(columns=["ip_risk_score"], inplace=True)

    if "amount_vs_avg_ratio" in df.columns and "ratio_q90" in params:
        df["log_amount_vs_avg_ratio"] = safe_log(df["amount_vs_avg_ratio"])
        df["high_amount_vs_avg_flag"] = (df["amount_vs_avg_ratio"] > params["ratio_q90"]).astype(int)
        new_var += ["log_amount_vs_avg_ratio", "high_amount_vs_avg_flag"]
        df.drop(columns=["amount_vs_avg_ratio"], inplace=True)

    if "velocity_1h" in df.columns and "velocity_q90" in params:
        df["log_velocity_1h"] = safe_log(df["velocity_1h"])
        df["high_velocity_1h_flag"] = (df["velocity_1h"] > params["velocity_q90"]).astype(int)
        new_var += ["log_velocity_1h", "high_velocity_1h_flag"]
        df.drop(columns=["velocity_1h"], inplace=True)

    if "credit_limit" in df.columns and "util_q90" in params and "amount" in df.columns:
        df["utilization"] = safe_divide(df["amount"], df["credit_limit"])
        df["high_utilization_flag"] = (df["utilization"] > params["util_q90"]).astype(int)
        new_var += ["utilization", "high_utilization_flag"]
        df.drop(columns=["amount"], inplace=True)
        df.drop(columns=["credit_limit"], inplace=True)

    return df, new_var


def fit_onehot(df: pd.DataFrame, cate_cols: list[str]) -> tuple[pd.DataFrame, OneHotEncoder]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Only keep cate cols that exist
    cate_cols = [c for c in cate_cols if c in df.columns]
    if not cate_cols:
        return df, encoder

    encoded = encoder.fit_transform(df[cate_cols])
    feature_names = encoder.get_feature_names_out(cate_cols)

    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

    df = df.drop(columns=cate_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoder


def transform_onehot(df: pd.DataFrame, cate_cols: list[str], encoder: OneHotEncoder) -> pd.DataFrame:
    cate_cols = [c for c in cate_cols if c in df.columns]
    if not cate_cols:
        return df

    encoded = encoder.transform(df[cate_cols])
    feature_names = encoder.get_feature_names_out(cate_cols)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

    df = df.drop(columns=cate_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df


def scale_numeric(df: pd.DataFrame, num_cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()

    num_cols = [c for c in num_cols if c in df.columns]
    scaler = StandardScaler()

    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler


def label_encode(
    df: pd.DataFrame,
    cate_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()

    cate_cols = [c for c in cate_cols if c in df.columns]

    if encoders is None:
        encoders = {}
        for col in cate_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cate_cols:
            if col in encoders:
                le = encoders[col]
                s = df[col].astype(str)

                # map unseen to UNKNOWN
                s = s.map(lambda x: x if x in le.classes_ else "UNKNOWN")
                if "UNKNOWN" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "UNKNOWN")

                df[col] = le.transform(s)
            else:
                df[col] = 0

    return df, encoders


def save_parquet(df: pd.DataFrame, path: str) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()

    # Load
    train = load_table(args.train_input)
    test = load_table(args.test_input)

    # Prepare exactly like notebook
    train = basic_prepare(train)
    test = basic_prepare(test)

    # Fit params on train
    params = fit_params(train)

    # ✅ Save FE params to a predictable location (default: models/trained/fe_params.pkl)
    ensure_parent(args.fe_params_output)
    joblib.dump(params, args.fe_params_output)

    # ---------------------------
    # Logistic branch
    # ---------------------------
    train_log = train.copy()
    train_log, _ = add_features(train_log, params)

    train_log, encoder = fit_onehot(train_log, CATE_COLS)

    scale_cols = [c for c in NUM_COLS if c in train_log.columns]
    train_log, scaler = scale_numeric(train_log, scale_cols)

    train_columns = train_log.columns.tolist()

    test_log = test.copy()
    test_log, _ = add_features(test_log, params)
    test_log = transform_onehot(test_log, CATE_COLS, encoder)
    test_log = test_log.reindex(columns=train_columns, fill_value=0)

    if scale_cols:
        test_log[scale_cols] = scaler.transform(test_log[scale_cols])

    # ---------------------------
    # Tree branch
    # ---------------------------
    train_tree = train.copy()
    train_tree, _ = add_features(train_tree, params)
    train_tree, encoders = label_encode(train_tree, CATE_COLS)

    tree_columns = train_tree.columns.tolist()

    test_tree = test.copy()
    test_tree, _ = add_features(test_tree, params)
    test_tree, _ = label_encode(test_tree, CATE_COLS, encoders=encoders)
    test_tree = test_tree.reindex(columns=tree_columns, fill_value=0)

    # Save parquet exactly like notebook intent
    save_parquet(train_log, args.train_log_output)
    save_parquet(test_log, args.test_log_output)
    save_parquet(train_tree, args.train_tree_output)
    save_parquet(test_tree, args.test_tree_output)

    print("Feature engineering done.")
    print(f"Saved fe_params to: {args.fe_params_output}")


if __name__ == "__main__":
    main()