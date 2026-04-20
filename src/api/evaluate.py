"""
evaluate.py — Performance monitoring cho Fraud Detection model.

Flow:
  1. Load labeled predictions từ logger.load_labeled_predictions()
     (join predictions.jsonl với labeled_predictions.jsonl theo transaction_id)
  2. Tính: Precision, Recall, F1, ROC-AUC
  3. So sánh với baseline metrics trong models/trained/trained_model_meta.json
  4. Cập nhật Prometheus gauges
  5. Nếu F1 < threshold → trigger retraining

Gọi từ:
  - Endpoint GET /evaluate (on-demand)
  - Background scheduler sau mỗi N labeled samples
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_PATH = PROJECT_ROOT / "models" / "trained" / "trained_model_meta.json"

# Ngưỡng để trigger retraining
F1_DEGRADATION_THRESHOLD = 0.10    # F1 giảm > 10% so với baseline → trigger
MIN_F1_ABSOLUTE = 0.65             # F1 tuyệt đối < 0.65 → trigger dù không so sánh baseline
MIN_LABELED_SAMPLES = 50           # Cần ít nhất 50 samples để evaluate


def _load_baseline_metrics() -> dict:
    """Load baseline F1, Precision, Recall từ trained_model_meta.json."""
    if not META_PATH.exists():
        logger.warning("Không tìm thấy trained_model_meta.json — sẽ không so sánh baseline.")
        return {}

    try:
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        # Lấy metrics từ meta (tùy structure của tune_model.py)
        return {
            "precision": meta.get("best_precision", meta.get("precision")),
            "recall": meta.get("best_recall", meta.get("recall")),
            "f1": meta.get("best_f1", meta.get("f1")),
            "roc_auc": meta.get("best_roc_auc", meta.get("roc_auc")),
        }
    except Exception as exc:
        logger.warning("Không đọc được meta: %s", exc)
        return {}


def run_performance_evaluation(labeled_records: list[dict]) -> dict[str, Any]:
    """
    Tính performance metrics từ danh sách labeled records.

    Args:
        labeled_records: list dict từ logger.load_labeled_predictions()
                         Mỗi record cần có: is_fraud (predicted), actual_is_fraud, fraud_score

    Returns:
        dict với keys:
          - precision, recall, f1, roc_auc: float
          - baseline_*: float (từ meta)
          - degraded: bool
          - degradation_reason: str | None
          - sample_count: int
          - error: str | None
    """
    result: dict[str, Any] = {
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "baseline_precision": None,
        "baseline_recall": None,
        "baseline_f1": None,
        "baseline_roc_auc": None,
        "degraded": False,
        "degradation_reason": None,
        "sample_count": len(labeled_records),
        "error": None,
    }

    if len(labeled_records) < MIN_LABELED_SAMPLES:
        result["error"] = (
            f"Chưa đủ labeled samples ({len(labeled_records)}/{MIN_LABELED_SAMPLES}). "
            "Gửi ground-truth labels qua POST /feedback để enable performance monitoring."
        )
        logger.info(result["error"])
        return result

    try:
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_true = [r["actual_is_fraud"] for r in labeled_records]
        y_pred = [r["is_fraud"] for r in labeled_records]          # predicted label
        y_score = [r.get("fraud_score", 0.5) for r in labeled_records]  # probability

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except ValueError:
            # Không tính được nếu chỉ có 1 class
            roc_auc = None
            logger.warning("ROC-AUC không tính được (chỉ có 1 class trong y_true).")

        result.update({
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        })

        # So sánh với baseline
        baseline = _load_baseline_metrics()
        baseline_f1 = baseline.get("f1")
        result["baseline_precision"] = baseline.get("precision")
        result["baseline_recall"] = baseline.get("recall")
        result["baseline_f1"] = baseline_f1
        result["baseline_roc_auc"] = baseline.get("roc_auc")

        # Kiểm tra degradation
        reasons = []
        if f1 < MIN_F1_ABSOLUTE:
            reasons.append(f"F1={f1:.4f} < min threshold {MIN_F1_ABSOLUTE}")

        if baseline_f1 is not None:
            drop = baseline_f1 - f1
            if drop > F1_DEGRADATION_THRESHOLD:
                reasons.append(
                    f"F1 giảm {drop:.4f} so với baseline ({baseline_f1:.4f} → {f1:.4f})"
                )

        if reasons:
            result["degraded"] = True
            result["degradation_reason"] = "; ".join(reasons)
            logger.warning("Model degradation detected: %s", result["degradation_reason"])

    except ImportError:
        result["error"] = "scikit-learn chưa được cài."
        logger.error(result["error"])
    except Exception as exc:
        result["error"] = str(exc)
        logger.exception("Lỗi evaluate: %s", exc)

    return result


def update_performance_metrics(eval_result: dict[str, Any]) -> None:
    """
    Cập nhật Prometheus gauges từ evaluation result.
    Chỉ cập nhật nếu có giá trị hợp lệ (không None).
    """
    from .metrics import (
        LABELED_SAMPLES_GAUGE,
        MODEL_AUC_GAUGE,
        MODEL_F1_GAUGE,
        MODEL_PRECISION_GAUGE,
        MODEL_RECALL_GAUGE,
    )

    LABELED_SAMPLES_GAUGE.set(eval_result.get("sample_count", 0))

    if eval_result.get("precision") is not None:
        MODEL_PRECISION_GAUGE.set(eval_result["precision"])
    if eval_result.get("recall") is not None:
        MODEL_RECALL_GAUGE.set(eval_result["recall"])
    if eval_result.get("f1") is not None:
        MODEL_F1_GAUGE.set(eval_result["f1"])
    if eval_result.get("roc_auc") is not None:
        MODEL_AUC_GAUGE.set(eval_result["roc_auc"])
