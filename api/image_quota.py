from __future__ import annotations

import time
from threading import Lock, Thread
from typing import Any

from fastapi import APIRouter, Header, Query
from fastapi.concurrency import run_in_threadpool

from api.support import require_identity
from services.account_service import account_service


_REFRESH_DISABLED_VALUES = {"0", "false", "no", "off", "disabled", "cache", "cached"}
_REFRESH_SYNC_VALUES = {"sync", "wait", "blocking", "true-sync"}
_BACKGROUND_REFRESH_MIN_INTERVAL_SECONDS = 60.0

_refresh_lock = Lock()
_refresh_state: dict[str, Any] = {
    "running": False,
    "lastStartedAt": None,
    "lastFinishedAt": None,
    "attemptedCount": 0,
    "successCount": 0,
    "failureCount": 0,
    "lastError": None,
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _quota(account: dict[str, Any]) -> int:
    return max(0, _safe_int(account.get("quota")))


def _status(account: dict[str, Any]) -> str:
    return str(account.get("status") or "").strip()


def _image_quota_unknown(account: dict[str, Any]) -> bool:
    return bool(account.get("image_quota_unknown"))


def _is_active(account: dict[str, Any]) -> bool:
    return (
        _status(account) == "正常"
        and account.get("last_token_refresh_error_kind") != "reauth_required"
    )


def _is_image_available(account: dict[str, Any]) -> bool:
    if not _is_active(account):
        return False
    if account.get("last_token_refresh_error_kind") == "reauth_required":
        return False
    if str(account.get("source_type") or "").strip().lower() == "codex":
        codex_available = account.get("codex_available")
        if codex_available is not None:
            return bool(codex_available)
        return _quota(account) > 0
    if _image_quota_unknown(account):
        return True
    return _quota(account) > 0


def _normalize_refresh_mode(refresh: str | None) -> str:
    value = str(refresh or "").strip().lower()
    if value in _REFRESH_DISABLED_VALUES:
        return "disabled"
    if value in _REFRESH_SYNC_VALUES:
        return "sync"
    return "background"


def _access_tokens(accounts: list[dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for account in accounts:
        token = str(account.get("access_token") or account.get("accessToken") or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _begin_refresh() -> bool:
    with _refresh_lock:
        if bool(_refresh_state.get("running")):
            return False
        now = _now_ms()
        _refresh_state.update(
            {
                "running": True,
                "lastStartedAt": now,
                "lastFinishedAt": None,
                "attemptedCount": 0,
                "successCount": 0,
                "failureCount": 0,
                "lastError": None,
            }
        )
        return True


def _finish_refresh(*, attempted: int, success: int, failure: int, error: str | None = None) -> None:
    with _refresh_lock:
        _refresh_state.update(
            {
                "running": False,
                "lastFinishedAt": _now_ms(),
                "attemptedCount": attempted,
                "successCount": success,
                "failureCount": failure,
                "lastError": error,
            }
        )


def _refresh_accounts_once() -> None:
    accounts = account_service.list_accounts()
    tokens = _access_tokens(accounts)
    attempted = len(tokens)
    with _refresh_lock:
        _refresh_state["attemptedCount"] = attempted

    try:
        result = account_service.refresh_accounts(tokens)
    except Exception as exc:  # pragma: no cover - defensive path
        _finish_refresh(attempted=attempted, success=0, failure=attempted, error=str(exc))
        return

    errors = result.get("errors") if isinstance(result, dict) else []
    failure = len(errors) if isinstance(errors, list) else 0
    success = _safe_int(result.get("refreshed") if isinstance(result, dict) else 0)
    _finish_refresh(attempted=attempted, success=success, failure=failure)


def _start_background_refresh() -> tuple[bool, bool]:
    with _refresh_lock:
        if bool(_refresh_state.get("running")):
            return False, False
        last_started = _refresh_state.get("lastStartedAt")
        if isinstance(last_started, int) and _now_ms() - last_started < _BACKGROUND_REFRESH_MIN_INTERVAL_SECONDS * 1000:
            return False, True

    if not _begin_refresh():
        return False, False
    Thread(target=_refresh_accounts_once, name="image-quota-refresh", daemon=True).start()
    return True, False


def _run_sync_refresh() -> bool:
    if not _begin_refresh():
        return False
    _refresh_accounts_once()
    return True


def _refresh_snapshot(mode: str, *, queued: bool = False, throttled: bool = False) -> dict[str, Any]:
    with _refresh_lock:
        snapshot = dict(_refresh_state)
    snapshot["mode"] = mode
    snapshot["queued"] = bool(queued)
    snapshot["throttled"] = bool(throttled)
    return snapshot


def _quota_summary(refresh: dict[str, Any]) -> dict[str, Any]:
    accounts = account_service.list_accounts()
    active_accounts = [account for account in accounts if _is_active(account)]
    known_active_accounts = [account for account in active_accounts if not _image_quota_unknown(account)]
    total_remaining = sum(_quota(account) for account in known_active_accounts) if known_active_accounts else None
    unknown_quota_count = sum(1 for account in active_accounts if _image_quota_unknown(account))
    available_count = sum(1 for account in active_accounts if _is_image_available(account))
    zero_quota_count = sum(
        1
        for account in active_accounts
        if not _image_quota_unknown(account) and _quota(account) <= 0
    )
    limited_count = sum(1 for account in accounts if _status(account) == "限流")
    abnormal_count = sum(1 for account in accounts if _status(account) == "异常")
    disabled_count = sum(1 for account in accounts if _status(account) == "禁用")

    payload = {
        "object": "image_quota",
        "totalRemaining": total_remaining,
        "remaining": total_remaining,
        "totalRemainingKnown": total_remaining is not None,
        "accountCount": len(accounts),
        "snapshotCount": len(accounts),
        "activeCount": len(active_accounts),
        "availableCount": available_count,
        "unknownQuotaCount": unknown_quota_count,
        "zeroQuotaCount": zero_quota_count,
        "limitedCount": limited_count,
        "abnormalCount": abnormal_count,
        "disabledCount": disabled_count,
        "todayGeneratedImages": None,
        "totalSuccess": sum(_safe_int(account.get("success")) for account in accounts),
        "totalFailure": sum(_safe_int(account.get("fail")) for account in accounts),
        "lastRefreshedAt": refresh.get("lastFinishedAt"),
        "refresh": refresh,
    }
    payload.update(
        {
            "total_remaining": payload["totalRemaining"],
            "total_remaining_known": payload["totalRemainingKnown"],
            "account_count": payload["accountCount"],
            "available_count": payload["availableCount"],
            "unknown_quota_count": payload["unknownQuotaCount"],
            "today_generated_images": payload["todayGeneratedImages"],
            "last_refreshed_at": payload["lastRefreshedAt"],
        }
    )
    return payload


def _authorization_header(authorization: str | None, x_api_key: str | None) -> str | None:
    if authorization:
        return authorization
    if x_api_key:
        return f"Bearer {x_api_key}"
    return None


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/image-quota")
    @router.get("/v1/image-quota")
    @router.get("/images/quota")
    @router.get("/v1/images/quota")
    async def image_quota(
        refresh: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        require_identity(_authorization_header(authorization, x_api_key))
        mode = _normalize_refresh_mode(refresh)
        queued = False
        throttled = False
        if mode == "sync":
            await run_in_threadpool(_run_sync_refresh)
        elif mode == "background":
            queued, throttled = _start_background_refresh()
        refresh_info = _refresh_snapshot(mode, queued=queued, throttled=throttled)
        return await run_in_threadpool(_quota_summary, refresh_info)

    return router
