from __future__ import annotations

from typing import Any

from services.account_service import account_service
from services.model_service import model_catalog_service
from utils.helper import (
    CODEX_IMAGE_25_MODEL,
    CODEX_IMAGE_25_SUNBURST_MODEL,
    CODEX_IMAGE_MODEL,
    IMAGE_MODEL_PLAN_TYPES,
    WEB_IMAGE_25_MODEL,
    WEB_IMAGE_MODEL,
)


def list_models() -> dict[str, Any]:
    result = model_catalog_service.list_models()
    data = result.get("data")
    if not isinstance(data, list):
        return result
    seen = {str(item.get("id") or "").strip() for item in data if isinstance(item, dict)}
    dynamic_models: set[str] = set()
    accounts = account_service.list_accounts()
    web_types = {
        normalized
        for account in accounts
        if isinstance(account, dict)
           and account_service._normalize_source_type(account.get("source_type")) != "codex"
           and (normalized := account_service._normalize_account_type(account.get("type")))
    }
    codex_types = {
        normalized
        for account in accounts
        if isinstance(account, dict)
           and account_service._normalize_source_type(account.get("source_type")) == "codex"
           and (normalized := account_service._normalize_account_type(account.get("type")))
    }

    if web_types:
        dynamic_models.update({WEB_IMAGE_MODEL, WEB_IMAGE_25_MODEL})
    codex_base_models = (CODEX_IMAGE_MODEL, CODEX_IMAGE_25_MODEL, CODEX_IMAGE_25_SUNBURST_MODEL)
    codex_types_lower = {value.lower() for value in codex_types}
    if codex_types_lower & {"plus", "team", "pro"}:
        dynamic_models.update(codex_base_models)
    for plan_type in IMAGE_MODEL_PLAN_TYPES:
        if plan_type in codex_types_lower:
            dynamic_models.update(f"{plan_type}-{model}" for model in codex_base_models)

    for model in sorted(dynamic_models):
        if model not in seen:
            data.append({
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "chatgpt2api",
                "permission": [],
                "root": model,
                "parent": None,
            })
    return result
