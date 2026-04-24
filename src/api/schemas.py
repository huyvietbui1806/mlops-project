from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FraudDetectionRequest(BaseModel):
    # Nếu bạn muốn không bị "mất biến âm thầm", forbid extra fields.
    # (Nếu muốn mềm hơn, đổi thành extra="ignore")
    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    user_id: str  # bạn đang dùng user_id thay cho account_id

    # ===== numeric / boolean-ish fields from your dataset =====
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)

    amount: float = Field(..., gt=0)

    # dataset columns: card_present/device_known/is_foreign_txn/has_2fa are 0/1
    card_present: int = Field(..., ge=0, le=1)
    device_known: int = Field(..., ge=0, le=1)
    is_foreign_txn: int = Field(..., ge=0, le=1)
    has_2fa: int = Field(..., ge=0, le=1)

    # times/ages
    time_since_last_s: float = Field(..., ge=0)
    velocity_1h: float = Field(..., ge=0)
    amount_vs_avg_ratio: float = Field(..., ge=0)
    account_age_days: int = Field(..., ge=0)

    # credit
    credit_limit: float = Field(..., gt=0)

    # ===== categorical fields =====
    merchant_category: str
    merchant_country: str
    device_type: str

    # in your data it's like 4900, 5411...
    mcc_code: int = Field(..., ge=0)

    # optional: if you also want to accept ip_risk_score from request
    ip_risk_score: float = Field(..., ge=0)


class FraudResponse(BaseModel):
    is_fraud: bool
    fraud_score: float
    risk_level: Literal["Low", "Medium", "High"]
    triggered_rules: List[str]
    prediction_time: datetime


class BatchFraudResponse(BaseModel):
    results: List[FraudResponse]
    total: int
    fraud_count: int

class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_label: bool
    feedback_time: Optional[datetime] = None
    source: Literal["manual_review", "chargeback", "system", "other"] = "other"


class FeedbackResponse(BaseModel):
    status: str
    prediction_id: str
    stored_at: datetime