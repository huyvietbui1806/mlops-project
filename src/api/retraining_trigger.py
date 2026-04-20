"""
retraining_trigger.py — Trigger GitHub Actions CT pipeline khi model degrade.

Gọi GitHub REST API với workflow_dispatch để chạy ct.yaml.
Cần env var: GITHUB_PAT (Personal Access Token với quyền `workflow`).

Cách set:
  - Local: export GITHUB_PAT=ghp_xxx...
  - Docker: thêm vào environment section trong docker-compose.yaml
  - K8s: thêm vào secret và mount vào pod
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# =====================
# CONFIG từ environment
# =====================
GITHUB_PAT = os.getenv("GITHUB_PAT", "")
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "")       # e.g. "huyvietbui1806"
GITHUB_REPO = os.getenv("GITHUB_REPO", "mlops-project")
WORKFLOW_FILE = os.getenv("RETRAIN_WORKFLOW", "ct.yaml")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")


async def trigger_retraining(reason: str = "auto") -> dict:
    """
    Gọi GitHub Actions workflow_dispatch để khởi động CT pipeline.

    Args:
        reason: Lý do trigger — "drift" | "performance" | "manual" | "alert"

    Returns:
        dict với keys: triggered (bool), status_code (int), message (str)
    """
    result = {
        "triggered": False,
        "status_code": None,
        "message": "",
        "reason": reason,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    # Validate config
    if not GITHUB_PAT:
        result["message"] = (
            "GITHUB_PAT chưa được set. Thêm env var GITHUB_PAT=ghp_xxx để enable auto-retraining."
        )
        logger.warning(result["message"])
        return result

    if not GITHUB_OWNER:
        result["message"] = "GITHUB_OWNER chưa được set."
        logger.warning(result["message"])
        return result

    try:
        import httpx

        url = (
            f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
            f"/actions/workflows/{WORKFLOW_FILE}/dispatches"
        )
        headers = {
            "Authorization": f"token {GITHUB_PAT}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        payload = {
            "ref": GITHUB_BRANCH,
            "inputs": {
                "reason": reason,
            },
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        result["status_code"] = response.status_code

        if response.status_code == 204:
            result["triggered"] = True
            result["message"] = (
                f"CT pipeline triggered successfully. Reason: {reason}. "
                f"Check GitHub Actions: "
                f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions"
            )
            logger.info(result["message"])

            # Cập nhật Prometheus counter
            try:
                from .metrics import RETRAINING_TRIGGERED_TOTAL
                RETRAINING_TRIGGERED_TOTAL.labels(reason=reason).inc()
            except Exception:
                pass

        else:
            result["message"] = (
                f"GitHub API trả về {response.status_code}: {response.text[:200]}"
            )
            logger.error(result["message"])

    except ImportError:
        result["message"] = "httpx chưa được cài. Chạy: uv add httpx"
        logger.error(result["message"])
    except Exception as exc:
        result["message"] = f"Lỗi khi gọi GitHub API: {exc}"
        logger.exception(result["message"])

    return result


def maybe_trigger_retraining(
    *,
    drift_result: dict | None = None,
    eval_result: dict | None = None,
) -> str | None:
    """
    Kiểm tra điều kiện và quyết định có trigger retraining không.
    Trả về lý do nếu cần trigger, None nếu không cần.

    Dùng trong async context — caller phải await trigger_retraining() nếu kết quả != None.
    """
    reasons = []

    if drift_result:
        if drift_result.get("drift_detected"):
            score = drift_result.get("drift_score", 0)
            reasons.append(f"drift (score={score:.3f})")

    if eval_result:
        if eval_result.get("degraded"):
            reason_txt = eval_result.get("degradation_reason", "performance degraded")
            reasons.append(reason_txt)

    if reasons:
        return "; ".join(reasons)
    return None
