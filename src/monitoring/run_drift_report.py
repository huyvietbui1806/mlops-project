from google.cloud import bigquery
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.monitoring.report_store import GCS_BUCKET, upload_drift_report
import os

PROJECT_ID = os.getenv("PROJECT_ID", "")
DATASET = os.getenv("BQ_DATASET", "fraud_monitoring")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/raw/train.csv")

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
FROM `{PROJECT_ID}.{DATASET}.predictions_ext`
"""

client = bigquery.Client(project=PROJECT_ID)

reference_df = pd.read_csv(REFERENCE_DATA_PATH)
current_df = client.query(CURRENT_QUERY).to_dataframe()

common_cols = [c for c in current_df.columns if c in reference_df.columns]
reference_df = reference_df[common_cols]
current_df = current_df[common_cols]

for col in common_cols:
    try:
        reference_df[col] = reference_df[col].astype(current_df[col].dtype)
    except Exception:
        pass

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)

report.save_html("reports_monitoring/drift_report.html")
result = upload_drift_report("reports_monitoring/drift_report.html")

print("Drift report saved locally: reports_monitoring/drift_report.html")
print(f"Archive report uploaded to: gs://{GCS_BUCKET}/{result['archive_blob']}")
print(f"Latest report uploaded to: gs://{GCS_BUCKET}/{result['latest_blob']}")

print("Drift report saved to reports_monitoring/drift_report.html")