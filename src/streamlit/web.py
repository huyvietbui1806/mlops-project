"""
web.py — FastAPI client layer
Connects to: http://34.72.72.60

API response schema (actual):
{
    "is_fraud":         bool,
    "fraud_score":      float,       ← NOT "confidence"
    "risk_level":       str,         ← "High" / "Medium" / "Low"  (NOT "severity")
    "triggered_rules":  list[str],   ← NOT "risk_factors"
    "prediction_time":  str
}
"""

from __future__ import annotations
import requests
import logging

log = logging.getLogger(__name__)

BASE_URL = "http://136.111.173.2"
TIMEOUT  = 120  # seconds


# ── low-level helpers ──────────────────────────────────────────────────────

def _post(path: str, payload: dict) -> dict:
    url = f"{BASE_URL}{path}"
    log.debug("POST %s  payload=%s", url, payload)
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    log.debug("Response: %s", data)
    return data


def _get(path: str, params: dict = None) -> dict:
    url = f"{BASE_URL}{path}"
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ── field normalizer ───────────────────────────────────────────────────────
# Converts the real FastAPI response into a single canonical shape
# that panel.py always reads from. This is the SINGLE source of truth
# for field name mapping — change it here only.

import re

def _normalize(raw: dict) -> dict:
    rules = raw.get("triggered_rules", raw.get("risk_factors", []))
    normalized_rules = []

    for rule in rules:
        if isinstance(rule, str) and rule.strip():
            clean = re.sub(r"<[^>]+>", "", rule.strip())  # remove ALL HTML
            normalized_rules.append({
                "name": clean.replace("_", " ").title(),
                "detail": ""
            })

        elif isinstance(rule, dict):
            name = re.sub(r"<[^>]+>", "", str(rule.get("name", "Unknown")))
            detail = re.sub(r"<[^>]+>", "", str(rule.get("detail", "")))

            name = name.strip().replace("_", " ").title()
            detail = detail.strip()

            if name:
                normalized_rules.append({
                    "name": name,
                    "detail": detail
                })

    return {
        "is_fraud":         bool(raw.get("is_fraud", False)),
        "confidence":       float(raw.get("fraud_score", raw.get("confidence", 0.0))),
        "severity":         str(raw.get("risk_level", raw.get("severity", "Low"))).upper(),
        "risk_factors":     normalized_rules,
        "prediction_time":  raw.get("prediction_time", "—"),
        "origin_node":      raw.get("origin_node", "—"),
        "latency":          raw.get("latency", "—"),
        "_raw":             raw,
    }
# ── public API ─────────────────────────────────────────────────────────────

def analyze_transaction(tx_data: dict) -> dict:
    """
    POST /predict  — calls the real FastAPI endpoint.
    Returns normalized dict (use _normalize internally).

    Change the path below if your FastAPI route is different.
    Check http://34.72.72.60/docs to confirm the exact route.
    """
    # ↓ UPDATE THIS PATH to match your FastAPI route (check /docs)
    raw = _post("/predict", tx_data)
    return _normalize(raw)


def health_check() -> dict:
    """GET /health"""
    return _get("/health")


# ── safe wrapper (NO silent mock fallback) ─────────────────────────────────

def safe_analyze(tx_data: dict) -> "tuple[dict | None, str | None]":
    """
    Calls the real API.
    Returns (normalized_result, error_string_or_None).

    IMPORTANT: mock fallback is intentionally REMOVED.
    If the API is unreachable, we surface the real error so the user
    knows they are NOT seeing a real prediction.
    """
    try:
        result = analyze_transaction(tx_data)
        return result, None

    except requests.exceptions.ConnectionError:
        err = (
            f"❌ Cannot reach API at {BASE_URL}. "
            "Check that the FastAPI server is running and the URL is correct."
        )
    except requests.exceptions.Timeout:
        err = f"⏱ API timed out after {TIMEOUT}s. The server may be overloaded."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        body   = e.response.text[:300]
        err    = f"🔴 API returned HTTP {status}.\nResponse: {body}"
    except Exception as e:
        err = f"⚠️ Unexpected error: {type(e).__name__}: {e}"

    log.error("safe_analyze failed: %s", err)
    return None, err
