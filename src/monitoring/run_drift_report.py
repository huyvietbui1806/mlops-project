from google.cloud import bigquery
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset
from src.monitoring.report_store import GCS_BUCKET, upload_drift_report
import os

PROJECT_ID = os.getenv("PROJECT_ID", "")
DATASET = os.getenv("BQ_DATASET", "fraud_monitoring")
JOINED_VIEW = os.getenv(
    "JOINED_VIEW",
    f"{PROJECT_ID}.{DATASET}.prediction_feedback_joined"
)


CURRENT_QUERY = f"""
SELECT
  amount,
  hour_of_day,
  day_of_week,
  merchant_category,
  merchant_country,
  device_type,
  mcc_code,
  ip_risk_score,
  velocity_1h,
  amount_vs_avg_ratio,
  account_age_days,
  credit_limit,
  card_present,
  device_known,
  is_foreign_txn,
  has_2fa
FROM `{JOINED_VIEW}`
"""

client = bigquery.Client(project=PROJECT_ID)

reference_df = pd.read_csv(r"../../data/raw/train.csv")
current_df = client.query(CURRENT_QUERY).to_dataframe()

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)

report.save_html("reports_monitoring/drift_report.html")
result = upload_drift_report("reports/drift_report.html")

print("Drift report saved locally: reports/drift_report.html")
print(f"Archive report uploaded to: gs://{GCS_BUCKET}/{result['archive_blob']}")
print(f"Latest report uploaded to: gs://{GCS_BUCKET}/{result['latest_blob']}")

print("Drift report saved to reports/drift_report.html")