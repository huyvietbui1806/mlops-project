"""
drift.py — Data drift detection bằng Evidently.

Flow:
  1. Load reference data (data/raw/*.csv hoặc snapshot đã lưu)
  2. Load current data từ logs/predictions.jsonl (n records gần nhất)
  3. Chạy Evidently DataDriftPreset
  4. Lưu report HTML + JSON summary ra reports/drift/
  5. Cập nhật Prometheus gauges
  6. Nếu drift vượt ngưỡng → trigger retraining

Có thể gọi:
  - Từ endpoint GET /drift (on-demand)
  - Từ background scheduler (cron)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "drift"

# Evidently drift score threshold để trigger retraining
DRIFT_SCORE_THRESHOLD = 0.3   # > 30% features bị drift → alert
DRIFT_SHARE_THRESHOLD = 0.5   # share of drifted features > 50% → critical


# =====================
# FEATURE COLUMNS dùng để so sánh drift
# Khớp với FraudDetectionRequest (loại bỏ ID fields)
# =====================
NUMERIC_FEATURES = [
    "amount", "hour_of_day", "day_of_week", "is_weekend",
    "card_present", "device_known", "is_foreign_txn", "has_2fa",
    "time_since_last_s", "velocity_1h", "amount_vs_avg_ratio",
    "account_age_days", "credit_limit", "ip_risk_score",
]

CATEGORICAL_FEATURES = [
    "merchant_category", "merchant_country", "device_type", "mcc_code",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _load_reference_data() -> pd.DataFrame:
    """
    Load reference dataset từ data/raw/.
    Ưu tiên: drift_reference.csv → bất kỳ *.csv nào trong data/raw/.
    """
    # Thử load file reference chuyên dụng
    ref_path = DATA_DIR / "drift_reference.csv"
    if not ref_path.exists():
        # Fallback: dùng file csv đầu tiên tìm thấy
        csv_files = list(DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"Không tìm thấy reference data trong {DATA_DIR}. "
                "Hãy đặt file CSV vào data/raw/ hoặc tạo data/raw/drift_reference.csv."
            )
        ref_path = csv_files[0]
        logger.info("Dùng %s làm reference data", ref_path.name)

    df = pd.read_csv(ref_path)
    # Chỉ giữ lại các feature columns có trong schema
    available = [c for c in ALL_FEATURES if c in df.columns]
    return df[available]


def _predictions_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Chuyển list prediction records từ logger sang DataFrame.
    Flatten input_features ra columns.
    """
    rows = []
    for rec in records:
        row = {**rec.get("input_features", {})}
        # Giữ thêm metadata hữu ích
        row["fraud_score"] = rec.get("fraud_score")
        row["is_fraud"] = rec.get("is_fraud")
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    available = [c for c in ALL_FEATURES if c in df.columns]
    return df[available]


def run_drift_check(recent_predictions: list[dict]) -> dict[str, Any]:
    """
    Chạy Evidently DataDriftPreset.

    Args:
        recent_predictions: list dict từ logger.load_recent_predictions()

    Returns:
        dict với keys:
          - drift_detected: bool
          - drift_score: float (0-1)
          - drifted_features: list[str]
          - drifted_features_count: int
          - dataset_drift: bool
          - report_path: str | None
          - sample_size: int
          - error: str | None
    """
    result: dict[str, Any] = {
        "drift_detected": False,
        "drift_score": 0.0,
        "drifted_features": [],
        "drifted_features_count": 0,
        "dataset_drift": False,
        "report_path": None,
        "sample_size": len(recent_predictions),
        "error": None,
    }

    if len(recent_predictions) < 30:
        result["error"] = f"Không đủ dữ liệu ({len(recent_predictions)} records). Cần ít nhất 30."
        logger.warning(result["error"])
        return result

    try:
        # Import Evidently ở đây để tránh import lỗi nếu chưa cài
        from evidently import ColumnMapping
        from evidently.metric_presets import DataDriftPreset
        from evidently.report import Report

        ref_df = _load_reference_data()
        cur_df = _predictions_to_dataframe(recent_predictions)

        if cur_df.empty:
            result["error"] = "Không thể parse prediction records thành DataFrame."
            return result

        # Chỉ dùng các cột tồn tại ở cả hai
        common_cols = [c for c in ALL_FEATURES if c in ref_df.columns and c in cur_df.columns]
        ref_df = ref_df[common_cols]
        cur_df = cur_df[common_cols]

        column_mapping = ColumnMapping(
            numerical_features=[c for c in NUMERIC_FEATURES if c in common_cols],
            categorical_features=[c for c in CATEGORICAL_FEATURES if c in common_cols],
        )

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)

        # Lấy kết quả JSON
        report_dict = report.as_dict()
        metrics_data = report_dict.get("metrics", [])

        # Parse kết quả từ DataDriftPreset
        drifted_features: list[str] = []
        dataset_drift = False
        drift_share = 0.0

        for metric in metrics_data:
            metric_id = metric.get("metric", "")
            res = metric.get("result", {})

            if "DatasetDriftMetric" in metric_id:
                dataset_drift = res.get("dataset_drift", False)
                drift_share = res.get("drift_share", 0.0)
                n_drifted = res.get("number_of_drifted_columns", 0)
                result["dataset_drift"] = dataset_drift
                result["drift_score"] = round(drift_share, 4)
                result["drifted_features_count"] = n_drifted

            elif "ColumnDriftMetric" in metric_id:
                col_name = res.get("column_name", "")
                if res.get("drift_detected", False) and col_name:
                    drifted_features.append(col_name)

        result["drifted_features"] = drifted_features
        result["drift_detected"] = dataset_drift or (drift_share > DRIFT_SCORE_THRESHOLD)

        # Lưu HTML report
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = REPORT_DIR / f"drift_report_{ts}.html"
        report.save_html(str(report_path))
        result["report_path"] = str(report_path)
        logger.info("Drift report saved → %s", report_path)

        # Lưu JSON summary cạnh HTML
        summary_path = REPORT_DIR / f"drift_summary_{ts}.json"
        summary_path.write_text(
            json.dumps(result, default=str, indent=2), encoding="utf-8"
        )

    except ImportError:
        result["error"] = "evidently chưa được cài. Chạy: uv add evidently"
        logger.error(result["error"])
    except Exception as exc:
        result["error"] = str(exc)
        logger.exception("Lỗi khi chạy drift check: %s", exc)

    return result


def update_drift_metrics(drift_result: dict[str, Any]) -> None:
    """
    Cập nhật Prometheus gauges từ drift result.
    Import metrics ở đây để tránh circular import.
    """
    from .metrics import (
        DRIFT_DATASET_DRIFT_GAUGE,
        DRIFT_SCORE_GAUGE,
        DRIFTED_FEATURES_GAUGE,
    )

    DRIFT_SCORE_GAUGE.set(drift_result.get("drift_score", 0.0))
    DRIFTED_FEATURES_GAUGE.set(drift_result.get("drifted_features_count", 0))
    DRIFT_DATASET_DRIFT_GAUGE.set(1.0 if drift_result.get("dataset_drift") else 0.0)
