import io
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage


GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "")
PREDICTION_PREFIX = os.getenv("PREDICTION_PREFIX", "predictions")


_gcs_client: storage.Client | None = None


def _get_gcs_client() -> storage.Client:
    """Lazy-initialize and cache the GCS client."""
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def _build_gcs_blob_name(event_time: datetime, prediction_id: str) -> str:
    dt = event_time.strftime("%Y-%m-%d")
    hour = event_time.strftime("%H")
    return f"{PREDICTION_PREFIX}/dt={dt}/hour={hour}/{prediction_id}.parquet"


def save_prediction_record(record: dict) -> str:
    """
    Save 1 prediction record as 1 parquet object to GCS.
    Return GCS blob name (object path) if success.
    Raise exception if upload fails.
    """
    if not GCS_BUCKET:
        raise RuntimeError("Missing GCS_BUCKET_NAME environment variable")

    df = pd.DataFrame([record])

    buffer = io.BytesIO()
    df.to_parquet(
        buffer,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    buffer.seek(0)

    event_time = record["event_time"]
    if isinstance(event_time, str):
        event_time = datetime.fromisoformat(event_time)

    prediction_id = record["prediction_id"]
    blob_name = _build_gcs_blob_name(event_time, prediction_id)

    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(buffer, content_type="application/octet-stream")

    return blob_name


def make_prediction_record(
    *,
    request_id: str | None,
    transaction_id: str,
    user_id: str,
    request_payload: dict,
    response_payload: dict,
    model_version: str,
    model_type: str,
    dataset_branch: str,
) -> dict:
    event_time = datetime.now(timezone.utc)
    prediction_id = str(uuid.uuid4())

    return {
        "prediction_id": prediction_id,
        "request_id": request_id,
        "event_time": event_time.isoformat(),
        "transaction_id": transaction_id,
        "user_id": user_id,
        "model_version": model_version,
        "model_type": model_type,
        "dataset_branch": dataset_branch,
        "is_fraud": response_payload["is_fraud"],
        "fraud_score": response_payload["fraud_score"],
        "risk_level": response_payload["risk_level"],
        "triggered_rules": ",".join(response_payload.get("triggered_rules", [])),
        "prediction_time": response_payload["prediction_time"],
        # raw input fields
        **request_payload,
    }