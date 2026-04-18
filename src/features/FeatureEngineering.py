from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


TARGET = "is_fraud"

CATE_COLS = [
    "merchant_category",
    "merchant_country",
    "device_type",
    "mcc_code",
    "hour_of_day",
    "day_of_week",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature engineering for fraud detection with time-based train/valid split"
    )

    parser.add_argument("--train-input", type=str, default="data/raw/train.csv")
    parser.add_argument("--test-input", type=str, default="data/raw/test.csv")

    parser.add_argument("--valid-size", type=float, default=0.2)

    parser.add_argument("--train-log-output", type=str, default="data/processed/train_log.parquet")
    parser.add_argument("--valid-log-output", type=str, default="data/processed/valid_log.parquet")
    parser.add_argument("--test-log-output", type=str, default="data/processed/test_log.parquet")

    parser.add_argument("--train-tree-output", type=str, default="data/processed/train_tree.parquet")
    parser.add_argument("--valid-tree-output", type=str, default="data/processed/valid_tree.parquet")
    parser.add_argument("--test-tree-output", type=str, default="data/processed/test_tree.parquet")

    parser.add_argument("--fe-params-output", type=str, default="models/trained/fe_params.pkl")

    parser.add_argument("--onehot-encoder-output", type=str, default="models/artifacts/onehot_encoder.pkl")
    parser.add_argument("--scaler-output", type=str, default="models/artifacts/scaler.pkl")
    parser.add_argument("--label-encoders-output", type=str, default="models/artifacts/label_encoders.pkl")

    return parser.parse_args()


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_joblib(obj, path: str | Path) -> None:
    ensure_parent(path)
    joblib.dump(obj, path)


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

    if "timestamp" not in df.columns:
        raise ValueError("Expected 'timestamp' column for time-based split.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return df


def time_split_train_valid(df: pd.DataFrame, valid_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < valid_size < 1:
        raise ValueError("--valid-size must be between 0 and 1")

    n = len(df)
    if n < 2:
        raise ValueError("Not enough rows to split train/valid")

    split_idx = int(n * (1 - valid_size))
    split_idx = max(1, min(n - 1, split_idx))

    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()

    return train_df, valid_df


def fit_params(df: pd.DataFrame) -> dict:
    params: dict = {}

    if "amount" in df.columns:
        params["amount_q95"] = float(df["amount"].quantile(0.95))
    else:
        params["amount_q95"] = 0.0

    if "ip_risk_score" in df.columns:
        params["ip_q80"] = float(df["ip_risk_score"].quantile(0.80))

    if "amount_vs_avg_ratio" in df.columns:
        params["ratio_q90"] = float(df["amount_vs_avg_ratio"].quantile(0.90))

    if "velocity_1h" in df.columns:
        params["velocity_q90"] = float(df["velocity_1h"].quantile(0.90))

    if "credit_limit" in df.columns and "amount" in df.columns:
        util = df["amount"] / df["credit_limit"].replace(0, np.nan)
        params["util_q90"] = float(util.quantile(0.90))

    return params


def add_features(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    new_var: list[str] = []

    if "amount" in df.columns:
        df["log_amount"] = safe_log(df["amount"])
        df["is_round_amount"] = ((df["amount"] % 10 == 0) & (df["amount"] > 0)).astype(int)
        df["high_amount_flag"] = (df["amount"] > params.get("amount_q95", 0)).astype(int)
        new_var += ["log_amount", "is_round_amount", "high_amount_flag"]

    if "hour_of_day" in df.columns:
        df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
        df["is_business_hours"] = df["hour_of_day"].between(9, 17).astype(int)
        new_var += ["is_night", "is_business_hours"]

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


def remove_non_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=["timestamp"], errors="ignore")

    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if dt_cols:
        df = df.drop(columns=dt_cols)

    return df


def cast_categorical_to_str(df: pd.DataFrame, cate_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cate_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def fit_onehot(df: pd.DataFrame, cate_cols: list[str]) -> tuple[pd.DataFrame, OneHotEncoder, list[str]]:
    df = df.copy()
    cate_cols = [c for c in cate_cols if c in df.columns]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if not cate_cols:
        return df, encoder, []

    df = cast_categorical_to_str(df, cate_cols)

    encoded = encoder.fit_transform(df[cate_cols])
    feature_names = encoder.get_feature_names_out(cate_cols)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

    df = df.drop(columns=cate_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoder, cate_cols


def transform_onehot(df: pd.DataFrame, cate_cols: list[str], encoder: OneHotEncoder) -> pd.DataFrame:
    df = df.copy()
    cate_cols = [c for c in cate_cols if c in df.columns]

    if not cate_cols:
        return df

    df = cast_categorical_to_str(df, cate_cols)

    encoded = encoder.transform(df[cate_cols])
    feature_names = encoder.get_feature_names_out(cate_cols)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

    df = df.drop(columns=cate_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df


def scale_numeric(
    df: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    df = df.copy()
    num_cols = [c for c in num_cols if c in df.columns and c != TARGET]
    scaler = StandardScaler()

    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler, num_cols


def transform_numeric(df: pd.DataFrame, num_cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    num_cols = [c for c in num_cols if c in df.columns and c != TARGET]

    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])

    return df


def label_encode(
    df: pd.DataFrame,
    cate_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()

    cate_cols = [c for c in cate_cols if c in df.columns]
    df = cast_categorical_to_str(df, cate_cols)

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


def prepare_log_branch(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OneHotEncoder, StandardScaler]:
    train_log, _ = add_features(train_df.copy(), params)
    valid_log, _ = add_features(valid_df.copy(), params)
    test_log, _ = add_features(test_df.copy(), params)

    train_log = remove_non_model_columns(train_log)
    valid_log = remove_non_model_columns(valid_log)
    test_log = remove_non_model_columns(test_log)

    raw_cat_cols = [c for c in CATE_COLS if c in train_log.columns]
    raw_num_cols = [c for c in train_log.columns if c not in raw_cat_cols and c != TARGET]

    train_log, encoder, used_cat_cols = fit_onehot(train_log, raw_cat_cols)
    train_log, scaler, used_num_cols = scale_numeric(train_log, raw_num_cols)

    train_columns = train_log.columns.tolist()

    valid_log = transform_onehot(valid_log, used_cat_cols, encoder)
    valid_log = transform_numeric(valid_log, used_num_cols, scaler)
    valid_log = valid_log.reindex(columns=train_columns, fill_value=0)

    test_log = transform_onehot(test_log, used_cat_cols, encoder)
    test_log = transform_numeric(test_log, used_num_cols, scaler)
    test_log = test_log.reindex(columns=train_columns, fill_value=0)

    return train_log, valid_log, test_log, encoder, scaler


def prepare_tree_branch(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    train_tree, _ = add_features(train_df.copy(), params)
    valid_tree, _ = add_features(valid_df.copy(), params)
    test_tree, _ = add_features(test_df.copy(), params)

    train_tree = remove_non_model_columns(train_tree)
    valid_tree = remove_non_model_columns(valid_tree)
    test_tree = remove_non_model_columns(test_tree)

    train_tree, encoders = label_encode(train_tree, CATE_COLS)
    tree_columns = train_tree.columns.tolist()

    valid_tree, _ = label_encode(valid_tree, CATE_COLS, encoders=encoders)
    valid_tree = valid_tree.reindex(columns=tree_columns, fill_value=0)

    test_tree, _ = label_encode(test_tree, CATE_COLS, encoders=encoders)
    test_tree = test_tree.reindex(columns=tree_columns, fill_value=0)

    return train_tree, valid_tree, test_tree, encoders


def main() -> None:
    args = parse_args()

    full_train = load_table(args.train_input)
    test = load_table(args.test_input)

    full_train = basic_prepare(full_train)
    test = basic_prepare(test)

    train_df, valid_df = time_split_train_valid(full_train, valid_size=args.valid_size)

    params = fit_params(train_df)
    params["valid_size"] = float(args.valid_size)

    save_joblib(params, args.fe_params_output)

    train_log, valid_log, test_log, encoder, scaler = prepare_log_branch(train_df, valid_df, test, params)
    train_tree, valid_tree, test_tree, label_encoders = prepare_tree_branch(train_df, valid_df, test, params)

    save_parquet(train_log, args.train_log_output)
    save_parquet(valid_log, args.valid_log_output)
    save_parquet(test_log, args.test_log_output)

    save_parquet(train_tree, args.train_tree_output)
    save_parquet(valid_tree, args.valid_tree_output)
    save_parquet(test_tree, args.test_tree_output)

    save_joblib(encoder, args.onehot_encoder_output)
    save_joblib(scaler, args.scaler_output)
    save_joblib(label_encoders, args.label_encoders_output)

    print("Feature engineering done.")
    print(f"Train rows: {len(train_df)}")
    print(f"Valid rows: {len(valid_df)}")
    print(f"Test rows : {len(test)}")
    print(f"Saved fe_params to: {args.fe_params_output}")
    print(f"Saved onehot encoder to: {args.onehot_encoder_output}")
    print(f"Saved scaler to: {args.scaler_output}")
    print(f"Saved label encoders to: {args.label_encoders_output}")


if __name__ == "__main__":
    main()