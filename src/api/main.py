import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.inference import predict_fraud, batch_predict, ARTIFACTS
from src.api.schemas import FraudDetectionRequest, FraudResponse, BatchFraudResponse
from src.api.logger import get_logger

import os

from src.api.prediction_store import (
    make_prediction_record,
    save_prediction_record,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    required_keys = ["meta", "model", "fe_params", "model_columns"]
    missing = [k for k in required_keys if k not in ARTIFACTS]

    if missing:
        logger.error(
            "startup failed",
            extra={"missing_artifacts": missing},
        )
        raise RuntimeError(f"Startup failed — missing artifacts: {missing}")

    logger.info("application startup successful")
    yield
    logger.info("application shutdown")


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    request.state.request_id = request_id

    logger.info(
        "request started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception(
            "request failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "latency_ms": latency_ms,
            },
        )
        raise

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)

    logger.exception(
        "unhandled exception",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exc).__name__,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "path": str(request.url),
            "request_id": request_id,
        },
    )


@app.get("/")
def root():
    logger.info("root endpoint called")
    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    model_ok = (
        "model" in ARTIFACTS
        and "fe_params" in ARTIFACTS
        and "model_columns" in ARTIFACTS
    )
    meta = ARTIFACTS.get("meta", {})

    logger.info(
        "health check evaluated",
        extra={
            "model_loaded": model_ok,
            "model_type": meta.get("selected_model", "unknown"),
            "dataset_branch": meta.get("dataset_branch", "unknown"),
        },
    )

    return {
        "status": "healthy" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model_type": meta.get("selected_model", "unknown"),
        "dataset_branch": meta.get("dataset_branch", "unknown"),
    }


@app.post("/predict", response_model=FraudResponse)
def predict(req: FraudDetectionRequest, request: Request):
    logger.info(
        "predict endpoint invoked",
        extra={
            "transaction_id": req.transaction_id,
            "user_id": req.user_id,
        },
    )

    result = predict_fraud(req)

    meta = ARTIFACTS.get("meta", {})
    model_version = os.getenv("MODEL_VERSION", "unknown")
    request_id = getattr(request.state, "request_id", None)

    record = make_prediction_record(
        request_id=request_id,
        transaction_id=req.transaction_id,
        user_id=req.user_id,
        request_payload=req.model_dump(),
        response_payload=result.model_dump(mode="json"),
        model_version=model_version,
        model_type=meta.get("selected_model", "unknown"),
        dataset_branch=meta.get("dataset_branch", "unknown"),
    )

    try:
        gcs_blob = save_prediction_record(record)
        logger.info(
            "prediction record saved to gcs",
            extra={
                "prediction_id": record["prediction_id"],
                "gcs_blob": gcs_blob,
            },
        )
    except Exception as e:
        logger.exception(
            "failed to save prediction record",
            extra={
                "prediction_id": record["prediction_id"],
                "error": str(e),
            },
        )

    return result

@app.post("/batch", response_model=BatchFraudResponse)
def batch(reqs: list[FraudDetectionRequest], request: Request):
    logger.info(
        "batch endpoint invoked",
        extra={"batch_size": len(reqs)},
    )

    responses = batch_predict(reqs)

    meta = ARTIFACTS.get("meta", {})
    model_version = os.getenv("MODEL_VERSION", "unknown")
    request_id = getattr(request.state, "request_id", None)

    for req, result in zip(reqs, responses):
        record = make_prediction_record(
            request_id=request_id,
            transaction_id=req.transaction_id,
            user_id=req.user_id,
            request_payload=req.model_dump(),
            response_payload=result.model_dump(mode="json"),
            model_version=model_version,
            model_type=meta.get("selected_model", "unknown"),
            dataset_branch=meta.get("dataset_branch", "unknown"),
        )

        try:
            gcs_blob = save_prediction_record(record)
            logger.info(
                "batch prediction record saved to gcs",
                extra={
                    "prediction_id": record["prediction_id"],
                    "gcs_blob": gcs_blob,
                },
            )
        except Exception as e:
            logger.exception(
                "failed to save batch prediction record",
                extra={
                    "prediction_id": record["prediction_id"],
                    "error": str(e),
                },
            )

    return BatchFraudResponse(
        results=responses,
        total=len(responses),
        fraud_count=sum(1 for r in responses if r.is_fraud),
    )