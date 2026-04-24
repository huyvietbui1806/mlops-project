import io
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage


GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "")
FEEDBACK_PREFIX = os.getenv("FEEDBACK_PREFIX", "feedbacks")


_gcs_client: storage.Client | None = None


def _get_gcs_client() -> storage.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def _build_gcs_blob_name(event_time: datetime, feedback_id: str) -> str:
    dt = event_time.strftime("%Y-%m-%d")
    hour = event_time.strftime("%H")
    return f"{FEEDBACK_PREFIX}/dt={dt}/hour={hour}/{feedback_id}.parquet"


def save_feedback_record(record: dict) -> str:
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

    event_time = record["stored_at"]
    if isinstance(event_time, str):
        event_time = datetime.fromisoformat(event_time)

    feedback_id = record["feedback_id"]
    blob_name = _build_gcs_blob_name(event_time, feedback_id)

    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(buffer, content_type="application/octet-stream")

    return blob_name


def make_feedback_record(
    *,
    request_id: str | None,
    prediction_id: str,
    actual_label: bool,
    feedback_time: datetime | None,
    source: str,
) -> dict:
    stored_at = datetime.now(timezone.utc)
    feedback_id = str(uuid.uuid4())

    return {
        "feedback_id": feedback_id,
        "request_id": request_id,
        "prediction_id": prediction_id,
        "actual_label": actual_label,
        "feedback_time": feedback_time.isoformat() if feedback_time else None,
        "source": source,
        "stored_at": stored_at.isoformat(),
    }