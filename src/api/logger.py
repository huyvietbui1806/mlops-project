"""
logger.py — Prediction logger cho MLOps monitoring trên GKE.

Chiến lược lưu trữ:
  - Nếu LOG_S3_BUCKET được set → ghi THẲNG lên S3 (primary, không qua local)
  - Nếu không có S3               → fallback ghi local (cho dev/test)

Lý do: GKE pods là ephemeral — filesystem mất khi pod restart.
S3 là "source of truth" duy nhất trong production.

S3 layout:
  s3://<BUCKET>/<PREFIX>/predictions.jsonl
  s3://<BUCKET>/<PREFIX>/labeled_predictions.jsonl

Env vars:
  LOG_S3_BUCKET   : tên bucket  (bắt buộc để bật S3 mode)
  LOG_S3_PREFIX   : folder trong bucket  (default: "logs")
  AWS_REGION      : default "ap-southeast-1"
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  (hoặc GKE Workload Identity)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger("fraud_api.logger")

# =====================
# LOCAL PATHS (fallback)
# =====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"
LABELED_LOG = LOG_DIR / "labeled_predictions.jsonl"

_lock = threading.Lock()

# =====================
# S3 CONFIG
# =====================
_S3_BUCKET: str | None = os.getenv("LOG_S3_BUCKET")
_S3_PREFIX: str = os.getenv("LOG_S3_PREFIX", "logs").rstrip("/")
_AWS_REGION: str = os.getenv("AWS_REGION", "ap-southeast-1")

_S3_CLIENT = None
_s3_init_lock = threading.Lock()

# Tên file trên S3
_S3_PREDICTION_KEY = f"{_S3_PREFIX}/predictions.jsonl"
_S3_LABELED_KEY = f"{_S3_PREFIX}/labeled_predictions.jsonl"


def _is_s3_enabled() -> bool:
    return bool(_S3_BUCKET)


def _get_s3_client():
    """Lazy-init boto3 S3 client (singleton, thread-safe)."""
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT  # None hoặc False đều trả về ngay

    with _s3_init_lock:
        if _S3_CLIENT is None:
            try:
                import boto3
                _S3_CLIENT = boto3.client("s3", region_name=_AWS_REGION)
                _log.info(
                    "S3 logger ready — bucket=%s prefix=%s region=%s",
                    _S3_BUCKET, _S3_PREFIX, _AWS_REGION,
                )
            except Exception as e:
                _log.error("Cannot init S3 client: %s — falling back to local.", e)
                _S3_CLIENT = False  # disabled, không thử lại
    return _S3_CLIENT


# =====================
# S3 OPERATIONS
# =====================
def _s3_append_record(s3_key: str, record: dict) -> None:
    """
    Append 1 record vào JSONL file trên S3.

    Cơ chế: GetObject → append dòng mới → PutObject.
    Thread-safe nhờ _lock (tránh race condition khi nhiều pods).

    Lưu ý: Với traffic rất cao (>100 req/s), nên dùng SQS + Lambda consumer
    thay vì append trực tiếp. Với quy mô hiện tại (fraud detection) thì OK.
    """
    client = _get_s3_client()
    if not client:
        raise RuntimeError("S3 client unavailable")

    new_line = json.dumps(record, default=str) + "\n"

    with _lock:
        # Đọc nội dung hiện tại (nếu có)
        try:
            resp = client.get_object(Bucket=_S3_BUCKET, Key=s3_key)
            existing = resp["Body"].read().decode("utf-8")
        except client.exceptions.NoSuchKey:
            existing = ""
        except Exception:
            # Key chưa tồn tại (bắt lỗi chung vì tên exception khác nhau tuỳ boto3 version)
            existing = ""

        # Ghi lại toàn bộ nội dung + dòng mới
        updated = existing + new_line
        client.put_object(
            Bucket=_S3_BUCKET,
            Key=s3_key,
            Body=updated.encode("utf-8"),
            ContentType="application/x-ndjson",
        )


def _s3_read_lines(s3_key: str, n: int | None = None) -> list[str]:
    """Đọc n dòng cuối (hoặc toàn bộ) từ JSONL file trên S3."""
    client = _get_s3_client()
    if not client:
        return []

    try:
        resp = client.get_object(Bucket=_S3_BUCKET, Key=s3_key)
        content = resp["Body"].read().decode("utf-8").strip()
        lines = content.splitlines() if content else []
        return lines[-n:] if (n and len(lines) > n) else lines
    except Exception as e:
        _log.warning("S3 read failed for key=%s: %s", s3_key, e)
        return []


# =====================
# LOCAL OPERATIONS (fallback)
# =====================
def _local_append_record(path: Path, record: dict) -> None:
    """Append 1 record vào JSONL file local (thread-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def _local_read_lines(path: Path, n: int | None = None) -> list[str]:
    """Đọc n dòng cuối từ JSONL file local."""
    if not path.exists():
        return []
    with _lock:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    return lines[-n:] if (n and len(lines) > n) else lines


# =====================
# UNIFIED WRITE / READ
# =====================
def _append_record(s3_key: str, local_path: Path, record: dict) -> None:
    """
    Ghi record: S3 nếu có bucket, ngược lại ghi local.
    Lỗi S3 → thử fallback local để không mất data.
    """
    if _is_s3_enabled():
        try:
            _s3_append_record(s3_key, record)
            return
        except Exception as e:
            _log.warning("S3 write failed (%s) — fallback to local: %s", s3_key, e)

    # Fallback hoặc dev mode
    _local_append_record(local_path, record)


def _read_lines(s3_key: str, local_path: Path, n: int | None = None) -> list[str]:
    """Đọc từ S3 nếu có, ngược lại đọc local."""
    if _is_s3_enabled():
        return _s3_read_lines(s3_key, n)
    return _local_read_lines(local_path, n)


def _parse_jsonl(lines: list[str]) -> list[dict]:
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


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
    Ghi một prediction event.

    Production (GKE): ghi thẳng lên S3  →  s3://<BUCKET>/<PREFIX>/predictions.jsonl
    Development     : ghi local          →  logs/predictions.jsonl

    Schema mỗi dòng:
    {
        "timestamp": "2026-04-20T12:00:00+00:00",
        "transaction_id": "txn_001",
        "user_id": "usr_42",
        "input_features": { ... },
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
    _append_record(_S3_PREDICTION_KEY, PREDICTION_LOG, record)


def log_feedback(
    *,
    transaction_id: str,
    actual_is_fraud: bool,
    feedback_source: str = "manual",
) -> None:
    """
    Ghi ground-truth label từ POST /feedback.

    Production (GKE): s3://<BUCKET>/<PREFIX>/labeled_predictions.jsonl
    Development     : logs/labeled_predictions.jsonl
    """
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "transaction_id": transaction_id,
        "actual_is_fraud": actual_is_fraud,
        "feedback_source": feedback_source,
    }
    _append_record(_S3_LABELED_KEY, LABELED_LOG, record)


def load_recent_predictions(n: int = 500) -> list[dict]:
    """Đọc n predictions gần nhất (từ S3 hoặc local)."""
    lines = _read_lines(_S3_PREDICTION_KEY, PREDICTION_LOG, n)
    return _parse_jsonl(lines)


def load_labeled_predictions() -> list[dict]:
    """
    Join predictions với labeled_predictions theo transaction_id.
    Trả về list records có đủ predicted + actual label.
    """
    label_lines = _read_lines(_S3_LABELED_KEY, LABELED_LOG, n=None)
    if not label_lines:
        return []

    predictions = {r["transaction_id"]: r for r in load_recent_predictions(n=10_000)}

    labeled = []
    for fb in _parse_jsonl(label_lines):
        txn_id = fb.get("transaction_id")
        if txn_id and txn_id in predictions:
            labeled.append({**predictions[txn_id], **fb})

    return labeled
