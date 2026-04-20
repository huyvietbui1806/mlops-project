"""
main.py — FastAPI application với đầy đủ monitoring.

Endpoints:
  GET  /              — root info
  GET  /health        — health check (model + artifacts)
  GET  /metrics       — Prometheus metrics (auto-mounted by instrumentator)
  POST /predict       — single prediction + logging + metrics
  POST /batch         — batch prediction + logging + metrics
  POST /feedback      — nhận ground-truth labels cho performance monitoring
  GET  /drift         — chạy Evidently drift check on-demand
  GET  /evaluate      — chạy performance evaluation on-demand
  POST /alert-webhook — nhận alert từ Alertmanager → trigger retraining
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.inference import ARTIFACTS, batch_predict_raw, predict_fraud_raw
from src.api.schemas import (
    AlertWebhookPayload,
    BatchFraudResponse,
    DriftResponse,
    EvaluateResponse,
    FeedbackRequest,
    FraudDetectionRequest,
    FraudResponse,
)

# =====================
# LOGGING
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger("fraud_api")


# =====================
# LIFESPAN
# =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: validate artifacts + setup Prometheus instrumentator."""
    required_keys = ["meta", "model", "fe_params", "model_columns"]
    missing = [k for k in required_keys if k not in ARTIFACTS]
    if missing:
        raise RuntimeError(f"Startup failed — missing artifacts: {missing}")

    # Khởi động Prometheus instrumentator
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        _log.info("Prometheus instrumentator mounted at /metrics")
    except ImportError:
        _log.warning("prometheus-fastapi-instrumentator chưa cài — /metrics sẽ không khả dụng.")

    _log.info("Fraud Detection API started. Model: %s", ARTIFACTS.get("meta", {}).get("selected_model", "?"))
    yield
    _log.info("Fraud Detection API shutting down.")


# =====================
# APP
# =====================
app = FastAPI(
    title="Fraud Detection API",
    version="2.0.0",
    description="Fraud detection với đầy đủ MLOps monitoring: drift, performance, alerts.",
    lifespan=lifespan,
)


# =====================
# EXCEPTION HANDLER
# =====================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    _log.exception("Unhandled error at %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# =====================
# ROOT & HEALTH
# =====================
@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "monitoring": {
            "drift": "GET /drift",
            "evaluate": "GET /evaluate",
            "feedback": "POST /feedback",
        },
    }


@app.get("/health")
def health_check():
    model_ok = all(k in ARTIFACTS for k in ["model", "fe_params", "model_columns"])
    meta = ARTIFACTS.get("meta", {})
    return {
        "status": "healthy" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model_type": meta.get("selected_model", "unknown"),
        "dataset_branch": meta.get("dataset_branch", "unknown"),
    }


# =====================
# PREDICT — single
# =====================
@app.post("/predict", response_model=FraudResponse)
def predict(req: FraudDetectionRequest):
    t0 = time.perf_counter()

    result = predict_fraud_raw(req)

    latency_ms = (time.perf_counter() - t0) * 1000

    # --- Logging ---
    try:
        from src.api.logger import load_recent_predictions, log_prediction
        log_prediction(
            transaction_id=req.transaction_id,
            user_id=req.user_id,
            input_features=req.dict(exclude={"transaction_id", "user_id"}),
            fraud_score=result["fraud_score"],
            is_fraud=result["is_fraud"],
            risk_level=result["risk_level"],
            triggered_rules=result["triggered_rules"],
            latency_ms=latency_ms,
        )

        # Cập nhật rolling gauges (async-unsafe nhưng OK trong sync handler)
        from src.api.metrics import update_rolling_gauges
        recent = load_recent_predictions(n=100)
        update_rolling_gauges(recent)

    except Exception as log_exc:
        _log.warning("Log prediction failed (non-fatal): %s", log_exc)

    # --- Prometheus counters ---
    try:
        from src.api.metrics import (
            FRAUD_DETECTED_TOTAL,
            PREDICTION_LATENCY,
            PREDICTION_TOTAL,
        )
        PREDICTION_TOTAL.labels(endpoint="/predict").inc()
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(latency_ms / 1000)
        if result["is_fraud"]:
            FRAUD_DETECTED_TOTAL.labels(risk_level=result["risk_level"]).inc()
    except Exception as metric_exc:
        _log.warning("Metrics update failed (non-fatal): %s", metric_exc)

    return FraudResponse(**result)


# =====================
# PREDICT — batch
# =====================
@app.post("/batch", response_model=BatchFraudResponse)
def batch(reqs: List[FraudDetectionRequest]):
    t0 = time.perf_counter()

    raw_results = batch_predict_raw(reqs)

    latency_ms = (time.perf_counter() - t0) * 1000

    # --- Logging (mỗi item) ---
    try:
        from src.api.logger import load_recent_predictions, log_prediction
        from src.api.metrics import update_rolling_gauges

        for req, result in zip(reqs, raw_results):
            log_prediction(
                transaction_id=req.transaction_id,
                user_id=req.user_id,
                input_features=req.dict(exclude={"transaction_id", "user_id"}),
                fraud_score=result["fraud_score"],
                is_fraud=result["is_fraud"],
                risk_level=result["risk_level"],
                triggered_rules=result["triggered_rules"],
                latency_ms=latency_ms / max(len(reqs), 1),
            )

        recent = load_recent_predictions(n=100)
        update_rolling_gauges(recent)

    except Exception as log_exc:
        _log.warning("Log batch failed (non-fatal): %s", log_exc)

    # --- Prometheus ---
    try:
        from src.api.metrics import (
            FRAUD_DETECTED_TOTAL,
            PREDICTION_LATENCY,
            PREDICTION_TOTAL,
        )
        PREDICTION_TOTAL.labels(endpoint="/batch").inc(len(raw_results))
        PREDICTION_LATENCY.labels(endpoint="/batch").observe(latency_ms / 1000)
        for r in raw_results:
            if r["is_fraud"]:
                FRAUD_DETECTED_TOTAL.labels(risk_level=r["risk_level"]).inc()
    except Exception as metric_exc:
        _log.warning("Metrics batch update failed (non-fatal): %s", metric_exc)

    responses = [FraudResponse(**r) for r in raw_results]
    return BatchFraudResponse(
        results=responses,
        total=len(responses),
        fraud_count=sum(1 for r in responses if r.is_fraud),
    )


# =====================
# FEEDBACK — nhận ground-truth labels
# =====================
@app.post("/feedback", status_code=201)
def submit_feedback(payload: FeedbackRequest):
    """
    Nhận ground-truth label cho một transaction.
    evaluate.py sẽ dùng labels này để tính F1, Precision, Recall.

    Ví dụ payload:
    {
        "transaction_id": "txn_001",
        "actual_is_fraud": true,
        "feedback_source": "chargeback"
    }
    """
    try:
        from src.api.logger import log_feedback
        log_feedback(
            transaction_id=payload.transaction_id,
            actual_is_fraud=payload.actual_is_fraud,
            feedback_source=payload.feedback_source or "manual",
        )
        return {
            "status": "recorded",
            "transaction_id": payload.transaction_id,
            "actual_is_fraud": payload.actual_is_fraud,
        }
    except Exception as exc:
        _log.exception("Feedback logging failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


# =====================
# DRIFT — on-demand Evidently check
# =====================
@app.get("/drift", response_model=DriftResponse)
def drift_check(n: int = 500):
    """
    Chạy Evidently drift check trên n records gần nhất.

    Query param:
      n (int): số predictions gần nhất dùng để check. Default=500.
    """
    from src.api.drift import run_drift_check, update_drift_metrics
    from src.api.logger import load_recent_predictions
    from src.api.retraining_trigger import maybe_trigger_retraining

    recent = load_recent_predictions(n=n)
    result = run_drift_check(recent)

    # Cập nhật Prometheus
    update_drift_metrics(result)

    # Kiểm tra có cần trigger retraining không
    trigger_reason = maybe_trigger_retraining(drift_result=result)
    if trigger_reason:
        _log.warning("Auto-retraining check: %s — trigger async...", trigger_reason)
        # Không await ở đây (sync endpoint) — log warning thôi
        # Để trigger async, dùng BackgroundTasks hoặc async endpoint
        _log.info("Tip: dùng async endpoint hoặc background task để trigger retraining")

    return DriftResponse(**result)


# =====================
# EVALUATE — on-demand performance check
# =====================
@app.get("/evaluate", response_model=EvaluateResponse)
def evaluate():
    """
    Tính performance metrics (F1, Precision, Recall, ROC-AUC)
    từ các predictions đã có ground-truth labels.

    Yêu cầu: đã có ít nhất 50 labeled samples qua POST /feedback.
    """
    from src.api.evaluate import run_performance_evaluation, update_performance_metrics
    from src.api.logger import load_labeled_predictions
    from src.api.retraining_trigger import maybe_trigger_retraining

    labeled = load_labeled_predictions()
    result = run_performance_evaluation(labeled)

    update_performance_metrics(result)

    trigger_reason = maybe_trigger_retraining(eval_result=result)
    if trigger_reason:
        _log.warning("Performance degradation: %s", trigger_reason)

    return EvaluateResponse(**result)


# =====================
# ALERT WEBHOOK — Alertmanager → retraining trigger
# =====================
@app.post("/alert-webhook", status_code=200)
async def alert_webhook(payload: AlertWebhookPayload):
    """
    Nhận alert từ Alertmanager.
    Nếu alert đang 'firing' → trigger retraining qua GitHub Actions.
    """
    from src.api.retraining_trigger import trigger_retraining

    if payload.status == "firing" and payload.alerts:
        alert_names = [a.get("labels", {}).get("alertname", "unknown") for a in payload.alerts]
        reason = f"alert:{','.join(alert_names)}"
        _log.warning("Alertmanager alert received: %s", reason)

        trigger_result = await trigger_retraining(reason=reason)
        return {
            "alert_received": True,
            "alert_names": alert_names,
            "retraining_triggered": trigger_result.get("triggered"),
            "message": trigger_result.get("message"),
        }

    return {
        "alert_received": True,
        "status": payload.status,
        "action": "no-op (not firing or no alerts)",
    }