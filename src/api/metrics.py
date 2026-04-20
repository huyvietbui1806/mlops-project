"""
metrics.py — Prometheus custom metrics cho Fraud Detection API.

Được expose qua endpoint /metrics (do prometheus-fastapi-instrumentator mount).
Grafana scrape từ Prometheus → dashboard.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# =====================
# REQUEST COUNTERS
# =====================
PREDICTION_TOTAL = Counter(
    name="fraud_prediction_total",
    documentation="Tổng số prediction requests đã xử lý",
    labelnames=["endpoint"],  # /predict | /batch
)

FRAUD_DETECTED_TOTAL = Counter(
    name="fraud_detected_total",
    documentation="Tổng số transactions bị flag là fraud",
    labelnames=["risk_level"],  # Low | Medium | High
)

PREDICTION_ERRORS_TOTAL = Counter(
    name="fraud_prediction_errors_total",
    documentation="Tổng số lỗi xảy ra khi predict",
    labelnames=["error_type"],
)

# =====================
# LATENCY HISTOGRAM
# =====================
PREDICTION_LATENCY = Histogram(
    name="fraud_prediction_latency_seconds",
    documentation="Thời gian xử lý mỗi prediction request (seconds)",
    labelnames=["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# =====================
# MODEL HEALTH GAUGES
# =====================
FRAUD_SCORE_GAUGE = Gauge(
    name="fraud_score_recent_avg",
    documentation="Average fraud score của 100 requests gần nhất",
)

FRAUD_RATE_GAUGE = Gauge(
    name="fraud_rate_recent",
    documentation="Tỷ lệ fraud (0-1) trong 100 requests gần nhất",
)

# =====================
# DRIFT GAUGES (cập nhật bởi drift.py)
# =====================
DRIFT_SCORE_GAUGE = Gauge(
    name="model_drift_score",
    documentation="Evidently data drift score (0=no drift, 1=full drift)",
)

DRIFTED_FEATURES_GAUGE = Gauge(
    name="model_drifted_features_count",
    documentation="Số features bị detect drift trong lần check gần nhất",
)

DRIFT_DATASET_DRIFT_GAUGE = Gauge(
    name="model_dataset_drift",
    documentation="1 nếu dataset-level drift được detect, 0 nếu không",
)

# =====================
# PERFORMANCE GAUGES (cập nhật bởi evaluate.py)
# =====================
MODEL_PRECISION_GAUGE = Gauge(
    name="model_precision",
    documentation="Precision của model trên labeled samples gần nhất",
)

MODEL_RECALL_GAUGE = Gauge(
    name="model_recall",
    documentation="Recall của model trên labeled samples gần nhất",
)

MODEL_F1_GAUGE = Gauge(
    name="model_f1_score",
    documentation="F1 score của model trên labeled samples gần nhất",
)

MODEL_AUC_GAUGE = Gauge(
    name="model_roc_auc",
    documentation="ROC-AUC của model trên labeled samples gần nhất",
)

LABELED_SAMPLES_GAUGE = Gauge(
    name="model_labeled_samples_count",
    documentation="Số labeled samples dùng để tính performance metrics",
)

# =====================
# RETRAINING GAUGE
# =====================
RETRAINING_TRIGGERED_TOTAL = Counter(
    name="retraining_triggered_total",
    documentation="Số lần retraining được trigger",
    labelnames=["reason"],  # drift | performance | manual
)


# =====================
# HELPER: cập nhật rolling gauges từ danh sách predictions gần đây
# =====================
def update_rolling_gauges(recent_predictions: list[dict]) -> None:
    """
    Tính toán và cập nhật fraud_score_recent_avg + fraud_rate_recent
    dựa trên danh sách predictions gần nhất.

    Gọi từ main.py sau mỗi request (async-safe vì Gauge.set() là thread-safe).
    """
    if not recent_predictions:
        return

    scores = [r.get("fraud_score", 0.0) for r in recent_predictions]
    frauds = [r.get("is_fraud", False) for r in recent_predictions]

    avg_score = sum(scores) / len(scores)
    fraud_rate = sum(frauds) / len(frauds)

    FRAUD_SCORE_GAUGE.set(avg_score)
    FRAUD_RATE_GAUGE.set(fraud_rate)
