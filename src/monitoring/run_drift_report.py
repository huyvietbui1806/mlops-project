import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

from src.monitoring.report_store import GCS_BUCKET, upload_drift_report

# ==============================
# ENV CONFIG
# ==============================
PROJECT_ID = os.getenv("PROJECT_ID", "")
DATASET = os.getenv("BQ_DATASET", "fraud_monitoring")
JOINED_VIEW = os.getenv(
    "JOINED_VIEW",
    f"{PROJECT_ID}.{DATASET}.prediction_feedback_joined"
)

REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/raw/train.csv")

# ==============================
# QUERY
# ==============================
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

# ==============================
# LOAD DATA
# ==============================
client = bigquery.Client(project=PROJECT_ID)

print("Loading reference data...")
reference_df = pd.read_csv(REFERENCE_DATA_PATH)

print("Loading current data from BigQuery...")
current_df = client.query(CURRENT_QUERY).to_dataframe()

print("Reference shape:", reference_df.shape)
print("Current shape:", current_df.shape)

# ==============================
# ALIGN COLUMNS
# ==============================
common_cols = [c for c in current_df.columns if c in reference_df.columns]

reference_df = reference_df[common_cols].copy()
current_df = current_df[common_cols].copy()

print("Common columns:", len(common_cols))

# ==============================
# CLEAN DATA FUNCTION
# ==============================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

reference_df = preprocess(reference_df)
current_df = preprocess(current_df)

# ==============================
# ALIGN DTYPES
# ==============================
for col in common_cols:
    try:
        reference_df[col] = reference_df[col].astype(current_df[col].dtype)
    except Exception:
        pass

# ==============================
# SELECT NUMERIC COLUMNS SAFELY
# ==============================
numeric_cols = current_df.select_dtypes(include=["number"]).columns.tolist()

clean_numeric_cols = []

for col in numeric_cols:
    # skip all-null
    if current_df[col].isna().all():
        print(f"Drop {col}: all NULL")
        continue

    # skip constant
    if current_df[col].nunique(dropna=True) <= 1:
        print(f"Drop {col}: constant")
        continue

    clean_numeric_cols.append(col)

print("Final numeric columns:", clean_numeric_cols)

# ==============================
# HANDLE NaN
# ==============================
reference_df[clean_numeric_cols] = reference_df[clean_numeric_cols].fillna(0)
current_df[clean_numeric_cols] = current_df[clean_numeric_cols].fillna(0)

# ==============================
# COLUMN MAPPING
# ==============================
column_mapping = ColumnMapping()
column_mapping.numerical_features = clean_numeric_cols

# ==============================
# RUN DRIFT REPORT
# ==============================
print("Running Evidently report...")

report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=reference_df,
    current_data=current_df,
    column_mapping=column_mapping
)

# ==============================
# SAVE REPORT
# ==============================
output_path = "reports_monitoring/drift_report.html"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

report.save_html(output_path)

print(f"Report saved locally: {output_path}")

# ==============================
# UPLOAD TO GCS
# ==============================
result = upload_drift_report(output_path)

print(f"Archive report uploaded to: gs://{GCS_BUCKET}/{result['archive_blob']}")
print(f"Latest report uploaded to: gs://{GCS_BUCKET}/{result['latest_blob']}")