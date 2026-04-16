from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.api.inference import predict_fraud, batch_predict, ARTIFACTS   # FIX: import ARTIFACTS để check startup
from src.api.schemas import FraudDetectionRequest, FraudResponse, BatchFraudResponse

# FIX: dùng lifespan thay on_event (cách được khuyến nghị từ FastAPI 0.93+)
# Kiểm tra artifact load ngay khi boot — nếu lỗi thì fail sớm, rõ ràng
@asynccontextmanager
async def lifespan(app: FastAPI):
    required_keys = ["meta", "model", "fe_params", "model_columns"]
    missing = [k for k in required_keys if k not in ARTIFACTS]
    if missing:
        raise RuntimeError(f"Startup failed — missing artifacts: {missing}")
    yield   # server chạy ở đây


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


# FIX: global exception handler — trả lỗi rõ ràng thay vì crash 500 không message
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "path": str(request.url),
        },
    )


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    # FIX: kiểm tra thực tế thay vì hardcode True
    model_ok = (
        "model" in ARTIFACTS
        and "fe_params" in ARTIFACTS
        and "model_columns" in ARTIFACTS
    )
    meta = ARTIFACTS.get("meta", {})
    return {
        "status": "healthy" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model_type": meta.get("selected_model", "unknown"),
        "dataset_branch": meta.get("dataset_branch", "unknown"),
    }


@app.post("/predict", response_model=FraudResponse)
def predict(req: FraudDetectionRequest):
    return predict_fraud(req)


# FIX: thêm response_model rõ ràng, trả đủ thông tin thay vì list[float] thô
@app.post("/batch", response_model=BatchFraudResponse)
def batch(reqs: list[FraudDetectionRequest]):
    responses = batch_predict(reqs)
    return BatchFraudResponse(
        results=responses,
        total=len(responses),
        fraud_count=sum(1 for r in responses if r.is_fraud),
    )