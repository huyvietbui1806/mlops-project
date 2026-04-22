# MLOps Monitoring Stack — Implementation Plan

## Mục tiêu

Tích hợp đầy đủ vòng lặp monitoring vào project fraud-detection hiện tại mà **không phá vỡ** logic inference đang hoạt động. Cụ thể:

| Công cụ | Vai trò |
|---|---|
| **Request Logger** | Ghi raw input + prediction ra JSONL → nguồn dữ liệu cho Evidently |
| **Evidently** | Data drift detection (input features vs. reference data) |
| **`evaluate.py`** | Performance monitoring (precision, recall, F1 so với threshold) |
| **Prometheus** | Scrape metrics từ `/metrics` endpoint của API |
| **Grafana** | Dashboard visualize metrics từ Prometheus |
| **Alertmanager** | Alert khi model degrade (fraud rate bất thường, drift score cao) |
| **Retraining trigger** | Gọi GitHub Actions `workflow_dispatch` khi phát hiện degradation |

---

## User Review Required

> [!IMPORTANT]
> **Drift reference data**: Evidently cần một "reference dataset" để so sánh. Plan này dùng file `data/raw/` đã có. Nếu bạn muốn dùng file khác, hãy cho biết trước khi thực thi.

> [!IMPORTANT]
> **Ground-truth labels**: `evaluate.py` chỉ tính được performance khi có nhãn thực tế (`is_fraud` ground truth). Plan này log prediction ra file và chờ nhãn được cung cấp (batch delayed labels) — thực tế sẽ cần thêm một endpoint `POST /feedback` để nhận nhãn. Bạn có muốn thêm endpoint này không?

> [!WARNING]
> **Retraining trigger**: Trigger sẽ gọi GitHub Actions REST API với `workflow_dispatch`. Cần thêm secret `GITHUB_TOKEN` (personal access token với quyền `workflow`) vào môi trường chạy API. Secret này **khác** với `GITHUB_TOKEN` trong Actions (cái đó chỉ dùng được trong CI).

> [!NOTE]
> **Không cần cài thêm infrastructure phức tạp**: Prometheus + Grafana + Alertmanager sẽ chạy qua `kubernetes` — tách biệt hoàn toàn với app.

---

## Open Questions

1. Bạn muốn log lưu ở **local file** (JSONL) hay push lên một storage ngoài (S3, DB)?  
   → Push lên S3.
2. Drift report chạy **theo schedule** (cron) hay **on-demand** qua endpoint `/drift`?  
   → Plan mặc định: **cả hai** — cron job trong Docker + endpoint `/drift` để debug.
3. Retraining trigger: tự động hay **cần approve**?  
   → Plan mặc định: tự động gọi `workflow_dispatch`, CI vẫn là nơi thực thi thực tế.

---

## Proposed Changes

### 1. `src/api/` — Core API (refactor + thêm mới)

#### [MODIFY] [inference.py](file:///c:/Assigment/2025-2026/ML%20OPs/mlops-project/src/api/inference.py)
- Tách `predict_fraud()` trả về **raw dict** thay vì `FraudResponse` để `main.py` dễ log.
- Giữ nguyên toàn bộ logic preprocess, artifact loading.

#### [MODIFY] [main.py](file:///c:/Assigment/2025-2026/ML%20OPs/mlops-project/src/api/main.py)
- Thêm **Prometheus middleware** (`prometheus-fastapi-instrumentator`) — expose `/metrics`.
- Gọi `logger.log_prediction()` sau mỗi request `/predict` và `/batch`.
- Thêm endpoint `POST /feedback` để nhận ground-truth labels.
- Thêm endpoint `GET /drift` để trigger Evidently report on-demand.
- Thêm custom gauges: `fraud_score_avg`, `fraud_rate`, `drift_score`.

#### [NEW] [logger.py](file:///c:/Assigment/2025-2026/ML%20OPs/mlops-project/src/api/logger.py)
Ghi mỗi prediction ra `logs/predictions.jsonl`:
```json
{
  "timestamp": "2026-04-20T12:00:00",
  "transaction_id": "txn_123",
  "input_features": { ... },
  "fraud_score": 0.82,
  "is_fraud": true,
  "risk_level": "High",
  "triggered_rules": ["high_amount"]
}
```

#### [NEW] [metrics.py](file:///c:/Assigment/2025-2026/ML%20OPs/mlops-project/src/api/metrics.py)
Định nghĩa Prometheus custom metrics:
- `fraud_prediction_total` (Counter) — tổng số predictions
- `fraud_detected_total` (Counter) — số predictions là fraud
- `prediction_latency_seconds` (Histogram) — thời gian xử lý
- `fraud_score_gauge` (Gauge) — average fraud score (window 100 requests)
- `model_drift_score` (Gauge) — drift score từ Evidently (cập nhật sau mỗi lần chạy drift check)

#### [NEW] [drift.py](file:///c:/Assigment/2025-2026/ML%20OPs/mlops-project/src/api/drift.py)
Chạy Evidently `DataDriftPreset` so sánh log gần nhất vs. reference dataset:
- Input: `logs/predictions.jsonl` (current) + `data/raw/*.csv` (reference)
- Output: `reports/drift/drift_report_{timestamp}.html` + JSON summary
- Update `model_drift_score` gauge trong Prometheus
- Nếu drift score > threshold → trigger retraining

#### [NEW] [evaluate.py](file:///c:/Assigment/2025-2026/ML%20OPs/mlops-project/src/api/evaluate.py)
Performance monitoring:
- Load các predictions đã có ground-truth từ `logs/labeled_predictions.jsonl`
- Tính precision, recall, F1, AUC
- So sánh với `models/trained/trained_model_meta.json` (baseline metrics)
- Update Prometheus gauges: `model_precision`, `model_recall`, `model_f1`
- Nếu F1 < threshold → trigger retraining

#### [NEW] [retraining_trigger.py](file:///c:/Assigment/2025-2026/ML%20OPs/mlops-project/src/api/retraining_trigger.py)
Gọi GitHub Actions `workflow_dispatch` API:
```python
POST https://api.github.com/repos/{owner}/{repo}/actions/workflows/ct.yaml/dispatches
Authorization: token {GITHUB_PAT}
```
- Dùng `httpx` async
- Log trigger event ra file

---

### 2. `deployment/monitoring/` — Infrastructure (mới hoàn toàn)

#### [NEW] docker-compose.yaml
Orchestrate toàn bộ monitoring stack:
```
services:
  api          → FastAPI (port 8000)
  prometheus   → scrape /metrics (port 9090)
  grafana      → dashboard (port 3000)
  alertmanager → alerts (port 9093)
```

#### [NEW] prometheus/prometheus.yml
- Scrape `api:8000/metrics` mỗi 15s
- Alerting rules file: `alerts.yml`

#### [NEW] prometheus/alerts.yml
Alert rules:
| Alert | Condition | Severity |
|---|---|---|
| `ModelDriftHigh` | `model_drift_score > 0.3` for 5m | warning |
| `FraudRateAnomaly` | `fraud_rate > 0.5` for 5m | critical |
| `ModelF1Degraded` | `model_f1 < 0.7` for 10m | critical |
| `APIDown` | `up == 0` for 1m | critical |

#### [NEW] alertmanager/alertmanager.yml
- Route tất cả alert → webhook receiver
- Webhook gọi `POST /api/alert-webhook` trên FastAPI → trigger retraining

#### [NEW] grafana/provisioning/
- `datasources/prometheus.yml` — auto-provision Prometheus datasource
- `dashboards/fraud_detection.json` — dashboard với panels:
  - Request rate, Fraud rate, Avg fraud score
  - Prediction latency (p50, p95, p99)
  - Model performance metrics (F1, Precision, Recall)
  - Drift score timeline

---

### 3. `.github/workflows/ct.yaml` — Thêm trigger từ API

#### [MODIFY] [ct.yaml](file:///c:/Assigment/2025-2026/ML%20OPs/mlops-project/.github/workflows/ct.yaml)
Thêm trigger `workflow_dispatch` với input `reason`:
```yaml
workflow_dispatch:
  inputs:
    reason:
      description: 'Trigger reason (drift/performance/manual)'
      default: 'manual'
```

---

### 4. `pyproject.toml` — New dependencies

#### [MODIFY] [pyproject.toml](file:///c:/Assigment/2025-2026/ML%20OPs/mlops-project/pyproject.toml)
Thêm:
```
evidently>=0.4,<1
prometheus-fastapi-instrumentator>=7,<8
prometheus-client>=0.20,<1
httpx>=0.27,<1
```


### Manual
- Mở Grafana dashboard và confirm các panel hiển thị đúng
- Trigger manual retraining qua `/drift` endpoint và check GitHub Actions
