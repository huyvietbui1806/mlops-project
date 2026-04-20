"""
logger.py — Prediction logger cho MLOps monitoring.

Ghi mỗi prediction ra JSONL file để:
  - Evidently đọc → drift detection
  - evaluate.py đọc → performance monitoring (khi có ground-truth labels)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# =====================
# PATHS
# =====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"
LABELED_LOG = LOG_DIR / "labeled_predictions.jsonl"

# Thread-safe lock cho file writes
_lock = threading.Lock()


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, record: dict) -> None:
    """Append một record vào JSONL file (thread-safe)."""
    _ensure_log_dir()
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


# =====================
# PUBLIC API
# =====================
def log_prediction(
    *,
    transaction_id: str,
    user_id: str,
    input_features: dict[str, Any],
    fraud_score: float,
    is_fraud: bool,
    risk_level: str,
    triggered_rules: list[str],
    latency_ms: float,
) -> None:
    """
    Ghi một prediction event ra `logs/predictions.jsonl`.

    Mỗi dòng là một JSON object với schema:
    {
        "timestamp": "2026-04-20T12:00:00Z",
        "transaction_id": "txn_001",
        "user_id": "usr_42",
        "input_features": { ... },  # raw features từ request
        "fraud_score": 0.82,
        "is_fraud": true,
        "risk_level": "High",
        "triggered_rules": ["high_amount"],
        "latency_ms": 12.4
    }
    """
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "transaction_id": transaction_id,
        "user_id": user_id,
        "input_features": input_features,
        "fraud_score": fraud_score,
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "triggered_rules": triggered_rules,
        "latency_ms": round(latency_ms, 3),
    }
    _append_jsonl(PREDICTION_LOG, record)


def log_feedback(
    *,
    transaction_id: str,
    actual_is_fraud: bool,
    feedback_source: str = "manual",
) -> None:
    """
    Ghi ground-truth label từ endpoint POST /feedback ra `logs/labeled_predictions.jsonl`.
    evaluate.py sẽ join file này với predictions.jsonl để tính performance metrics.
    """
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "transaction_id": transaction_id,
        "actual_is_fraud": actual_is_fraud,
        "feedback_source": feedback_source,
    }
    _append_jsonl(LABELED_LOG, record)


def load_recent_predictions(n: int = 500) -> list[dict]:
    """
    Đọc n dòng gần nhất từ predictions.jsonl.
    Dùng bởi drift.py và evaluate.py.
    """
    if not PREDICTION_LOG.exists():
        return []

    with _lock:
        lines = PREDICTION_LOG.read_text(encoding="utf-8").strip().splitlines()

    recent = lines[-n:] if len(lines) > n else lines
    records = []
    for line in recent:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def load_labeled_predictions() -> list[dict]:
    """
    Join predictions.jsonl với labeled_predictions.jsonl theo transaction_id.
    Trả về list các records có cả predicted và actual label.
    """
    if not LABELED_LOG.exists():
        return []

    predictions = {r["transaction_id"]: r for r in load_recent_predictions(n=10_000)}

    labeled = []
    with _lock:
        lines = LABELED_LOG.read_text(encoding="utf-8").strip().splitlines()

    for line in lines:
        try:
            fb = json.loads(line)
        except json.JSONDecodeError:
            continue

        txn_id = fb.get("transaction_id")
        if txn_id and txn_id in predictions:
            merged = {**predictions[txn_id], **fb}
            labeled.append(merged)

    return labeled
