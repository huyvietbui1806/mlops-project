import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
# from evidently.metrics import DataDriftTable
# from evidently.metrics import DatasetDriftMetric

from src.monitoring.report_store import GCS_BUCKET, upload_drift_report

# ==============================
# ENV CONFIG
# ==============================
PROJECT_ID = os.getenv("PROJECT_ID", "")
DATASET = os.getenv("BQ_DATASET", "fraud_monitoring")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/raw/train.csv")
MIN_ROWS = int(os.getenv("MIN_ROWS", "50"))
MIN_UNIQUE = int(os.getenv("MIN_UNIQUE", "2"))

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
FROM `{PROJECT_ID}.{DATASET}.predictions_ext`
"""

# ==============================
# LOAD DATA
# ==============================
client = bigquery.Client(project=PROJECT_ID)

print("Loading reference data...")
reference_df = pd.read_csv(REFERENCE_DATA_PATH)
print("Reference shape:", reference_df.shape)

print("Loading current data from BigQuery...")
current_df = client.query(CURRENT_QUERY).to_dataframe()
print("Current shape:", current_df.shape)

# ==============================
# CHECK MINIMUM ROWS
# ==============================
if len(current_df) < MIN_ROWS:
    print(f"Not enough data: {len(current_df)} rows (min {MIN_ROWS}). Skipping report.")
    exit(0)

# ==============================
# ALIGN COLUMNS
# ==============================
common_cols = [c for c in current_df.columns if c in reference_df.columns]
print(f"Common columns: {len(common_cols)} — {common_cols}")

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
# FILTER COLUMNS ĐỦ ĐIỀU KIỆN
# ==============================
valid_cols = []

for col in common_cols:
    ref_col = reference_df[col]
    cur_col = current_df[col]

    if cur_col.isna().all() or ref_col.isna().all():
        print(f"Skip {col}: all NULL")
        continue

    if cur_col.nunique(dropna=True) < MIN_UNIQUE:
        print(f"Skip {col}: constant in current")
        continue

    if ref_col.nunique(dropna=True) < MIN_UNIQUE:
        print(f"Skip {col}: constant in reference")
        continue

    if cur_col.dropna().shape[0] < MIN_ROWS:
        print(f"Skip {col}: not enough rows ({cur_col.dropna().shape[0]})")
        continue
    
    valid_cols.append(col)

print(f"Valid columns: {valid_cols}")

if not valid_cols:
    print("No valid columns. Skipping report.")
    exit(0)

# ==============================
# APPLY VALID COLUMNS
# ==============================
reference_df = reference_df[valid_cols].copy()
current_df = current_df[valid_cols].copy()

# Fill NaN
reference_df = reference_df.fillna(0)
current_df = current_df.fillna(0)

# ==============================
# COLUMN MAPPING
# ==============================
numeric_cols = [c for c in valid_cols if current_df[c].dtype in ["float64", "int64", "int32", "float32"]]
cat_cols = [c for c in valid_cols if current_df[c].dtype == "object"]

column_mapping = ColumnMapping()
column_mapping.numerical_features = numeric_cols
column_mapping.categorical_features = cat_cols

print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {cat_cols}")

# ==============================
# RUN DRIFT REPORT
# ==============================
print("Running Evidently report...")

# report = Report(metrics=[
#     DatasetDriftMetric(),
#     DataDriftTable(),
# ])

report = Report(metrics=[DataDriftPreset(cat_stattest_threshold=0.05)])

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