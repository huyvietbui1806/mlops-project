from prometheus_client import Counter, Histogram


HTTP_REQUESTS_TOTAL = Counter(
    "fraud_api_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "fraud_api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)

PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total number of prediction results generated",
    ["endpoint"],
)

POSITIVE_PREDICTIONS_TOTAL = Counter(
    "fraud_positive_predictions_total",
    "Total number of positive fraud predictions",
    ["endpoint"],
)

FEEDBACK_TOTAL = Counter(
    "fraud_feedback_total",
    "Total number of feedback events received",
    ["source"],
)