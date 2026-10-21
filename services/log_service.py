from __future__ import annotations

import hashlib
import json
import itertools
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from services.config import DATA_DIR
from services.protocol.error_response import anthropic_error_response, openai_error_response
from utils.helper import anthropic_sse_stream, sse_json_stream
from utils.log import logger

LOG_TYPE_CALL = "call"
LOG_TYPE_ACCOUNT = "account"
INTERNAL_RESPONSE_KEYS = {"_account_email", "_conversation_id"}

# 单条日志序列化后的硬上限与字段截断长度：防止 data: URI 等巨型内容
# 把 logs.jsonl 撑到百 MB 级（2026-09 实测单条 7.2MB 导致日志页 6.8s 才返回）。
LOG_MAX_LINE_BYTES = 256 * 1024
LOG_DETAIL_STRING_LIMIT = 2000
# list() 从文件尾部反向扫描的最多字节数；更大的文件只看最近这段。
LOG_MAX_SCAN_BYTES = 64 * 1024 * 1024
# 自动瘦身：追加时若文件超过阈值，仅保留尾部内容，避免长期运行无限膨胀。
LOG_ROTATE_THRESHOLD_BYTES = 64 * 1024 * 1024
LOG_ROTATE_KEEP_BYTES = 16 * 1024 * 1024
LOG_ROTATE_CHECK_INTERVAL_SECS = 3600.0
LOG_READ_BLOCK_BYTES = 1024 * 1024


class LogService:
    def __init__(
        self,
        path: Path,
        *,
        max_line_bytes: int = LOG_MAX_LINE_BYTES,
        detail_string_limit: int = LOG_DETAIL_STRING_LIMIT,
        max_scan_bytes: int = LOG_MAX_SCAN_BYTES,
        rotate_threshold_bytes: int = LOG_ROTATE_THRESHOLD_BYTES,
        rotate_keep_bytes: int = LOG_ROTATE_KEEP_BYTES,
        rotate_check_interval_secs: float = LOG_ROTATE_CHECK_INTERVAL_SECS,
        read_block_bytes: int = LOG_READ_BLOCK_BYTES,
    ):
        self.path = path
        self.max_line_bytes = max(1024, int(max_line_bytes))
        self.detail_string_limit = max(64, int(detail_string_limit))
        self.max_scan_bytes = max(1024, int(max_scan_bytes))
        self.rotate_threshold_bytes = max(1024, int(rotate_threshold_bytes))
        self.rotate_keep_bytes = max(1024, int(rotate_keep_bytes))
        self.rotate_check_interval_secs = max(0.0, float(rotate_check_interval_secs))
        self.read_block_bytes = max(64, int(read_block_bytes))
        self._io_lock = threading.Lock()
        self._last_rotate_check = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _legacy_id(raw_line: str, line_number: int) -> str:
        # 旧格式行（无 id 字段）用内容哈希生成稳定 id：与行号无关，
        # 文件追加/瘦身后 id 保持不变，便于后台按 id 删除。
        payload = raw_line.encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()[:24]

    def _parse_line(self, raw_line: str, line_number: int) -> dict[str, Any] | None:
        try:
            item = json.loads(raw_line)
        except Exception:
            return None
        if not isinstance(item, dict):
            return None
        parsed = dict(item)
        parsed["id"] = str(parsed.get("id") or self._legacy_id(raw_line, line_number))
        return parsed

    @staticmethod
    def _serialize_item(item: dict[str, Any]) -> str:
        return json.dumps(item, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _truncate_deep(cls, value: Any, limit: int) -> Any:
        if isinstance(value, str):
            return value if len(value) <= limit else value[: limit - 1] + "…"
        if isinstance(value, dict):
            return {key: cls._truncate_deep(item, limit) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._truncate_deep(item, limit) for item in value]
        return value

    @staticmethod
    def _matches_filters(item: dict[str, Any], *, type: str = "", start_date: str = "", end_date: str = "") -> bool:
        t = str(item.get("time") or "")
        day = t[:10]
        if type and item.get("type") != type:
            return False
        if start_date and day < start_date:
            return False
        if end_date and day > end_date:
            return False
        return True

    def _serialize_capped(self, item: dict[str, Any]) -> str:
        line = self._serialize_item(item)
        if len(line.encode("utf-8")) <= self.max_line_bytes:
            return line
        trimmed = dict(item)
        trimmed["detail"] = self._truncate_deep(trimmed.get("detail"), self.detail_string_limit)
        trimmed["detail_truncated"] = True
        line = self._serialize_item(trimmed)
        if len(line.encode("utf-8")) <= self.max_line_bytes:
            return line
        trimmed["detail"] = {"truncated": True, "reason": "detail exceeds log line cap"}
        return self._serialize_item(trimmed)

    def add(self, type: str, summary: str = "", detail: dict[str, Any] | None = None, **data: Any) -> None:
        item = {
            "id": uuid4().hex,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": type,
            "summary": summary,
            "detail": detail or data,
        }
        line = self._serialize_capped(item)
        with self._io_lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
            self._rotate_if_needed_locked()

    def _rotate_if_needed_locked(self) -> None:
        now = time.monotonic()
        if now - self._last_rotate_check < self.rotate_check_interval_secs:
            return
        self._last_rotate_check = now
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size <= self.rotate_threshold_bytes:
            return
        keep = min(self.rotate_keep_bytes, size)
        with self.path.open("rb") as file:
            file.seek(size - keep)
            tail = file.read(keep)
        # 丢弃首个不完整行，从完整行开始保留
        newline_index = tail.find(b"\n")
        if newline_index >= 0:
            tail = tail[newline_index + 1:]
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_bytes(tail)
        os.replace(tmp_path, self.path)
        logger.info({
            "event": "log_file_rotated",
            "path": str(self.path),
            "previous_size": size,
            "kept_bytes": len(tail),
        })

    def list(self, type: str = "", start_date: str = "", end_date: str = "", limit: int = 200) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        try:
            file_size = self.path.stat().st_size
        except OSError:
            return []
        items: list[dict[str, Any]] = []
        scan_floor = max(0, file_size - self.max_scan_bytes)

        def _handle_line(raw_line: str) -> None:
            if len(items) >= limit:
                return
            item = self._parse_line(raw_line, 0)
            if item is None:
                return
            if not self._matches_filters(item, type=type, start_date=start_date, end_date=end_date):
                return
            items.append(item)

        with self._io_lock:
            with self.path.open("rb") as file:
                position = file_size
                carry = b""
                while position > scan_floor and len(items) < limit:
                    block_size = min(self.read_block_bytes, position - scan_floor)
                    position -= block_size
                    file.seek(position)
                    block = file.read(block_size) + carry
                    pieces = block.split(b"\n")
                    # pieces[0] 可能是残行（起点在更早的块里），留给下一轮拼接
                    carry = pieces[0]
                    complete = pieces[1:] if position > scan_floor else pieces
                    for raw_bytes in reversed(complete):
                        raw_line = raw_bytes.decode("utf-8", errors="ignore").strip()
                        if raw_line:
                            _handle_line(raw_line)
        return items

    def delete(self, ids: list[str]) -> dict[str, int]:
        target_ids = {str(item or "").strip() for item in ids if str(item or "").strip()}
        if not self.path.exists() or not target_ids:
            return {"removed": 0}
        with self._io_lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            kept_lines: list[str] = []
            removed = 0
            for line_number, raw_line in enumerate(lines):
                item = self._parse_line(raw_line, line_number)
                if item is None:
                    kept_lines.append(raw_line)
                    continue
                if str(item.get("id") or "") in target_ids:
                    removed += 1
                    continue
                kept_lines.append(self._serialize_item(item))
            content = "\n".join(kept_lines)
            if content:
                content += "\n"
            self.path.write_text(content, encoding="utf-8")
        return {"removed": removed}


log_service = LogService(DATA_DIR / "logs.jsonl")


def _collect_urls(value: object) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "url" and isinstance(item, str):
                # data: URI 是内联图片载荷（关闭图片保存时可达数 MB），不属于 URL，
                # 收进日志会把单条记录撑到 MB 级，这里直接跳过。
                if not item.startswith("data:"):
                    urls.append(item)
            elif key == "urls" and isinstance(item, list):
                urls.extend(str(url) for url in item if isinstance(url, str) and not str(url).startswith("data:"))
            else:
                urls.extend(_collect_urls(item))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_collect_urls(item))
    return urls


def _collect_account_emails(value: object) -> list[str]:
    emails: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"_account_email", "account_email"} and isinstance(item, str) and item.strip():
                emails.append(item.strip())
            else:
                emails.extend(_collect_account_emails(item))
    elif isinstance(value, list):
        for item in value:
            emails.extend(_collect_account_emails(item))
    return emails


def _collect_conversation_ids(value: object) -> list[str]:
    ids: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "_conversation_id" and isinstance(item, str) and item.strip():
                ids.append(item.strip())
            else:
                ids.extend(_collect_conversation_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.extend(_collect_conversation_ids(item))
    return ids


def _strip_internal_response_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_internal_response_fields(item)
            for key, item in value.items()
            if key not in INTERNAL_RESPONSE_KEYS
        }
    if isinstance(value, list):
        return [_strip_internal_response_fields(item) for item in value]
    return value


def _request_excerpt(text: object, limit: int = 1000) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _image_error_response(exc: Exception) -> JSONResponse:
    from services.protocol.conversation import public_image_error_message

    message = public_image_error_message(str(exc))
    if "no available image quota" in message.lower():
        return openai_error_response(
            {
                "error": {
                    "message": "no available image quota",
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_quota",
                }
            },
            429,
        )
    if hasattr(exc, "to_openai_error") and hasattr(exc, "status_code"):
        return JSONResponse(status_code=int(exc.status_code), content=exc.to_openai_error())
    return openai_error_response(message, 502)


def _protocol_error_response(exc: Exception, status_code: int, sse: str) -> JSONResponse:
    message = str(exc)
    if sse == "anthropic":
        return anthropic_error_response(message, status_code)
    return openai_error_response(message, status_code)


def _next_item(items):
    try:
        return True, next(items)
    except StopIteration:
        return False, None


@dataclass
class LoggedCall:
    identity: dict[str, object]
    endpoint: str
    model: str
    summary: str
    started: float = field(default_factory=time.time)
    request_text: str = ""
    request_shape: dict[str, int] | None = None

    async def run(self, handler, *args, sse: str = "openai"):
        from services.protocol.conversation import ImageGenerationError

        try:
            result = await run_in_threadpool(handler, *args)
        except ImageGenerationError as exc:
            self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""),
                     conversation_id=getattr(exc, "conversation_id", ""))
            return _image_error_response(exc)
        except HTTPException as exc:
            self.log("调用失败", status="failed", error=str(exc.detail))
            raise
        except Exception as exc:
            self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""))
            if self.endpoint.startswith("/v1/images"):
                return _image_error_response(exc)
            return _protocol_error_response(exc, 502, sse)

        if isinstance(result, dict):
            self.log("调用完成", result)
            response = dict(result)
            response.pop("_account_email", None)
            return response

        sender = anthropic_sse_stream if sse == "anthropic" else sse_json_stream
        try:
            has_first, first = await run_in_threadpool(_next_item, result)
        except ImageGenerationError as exc:
            self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""),
                     conversation_id=getattr(exc, "conversation_id", ""))
            return _image_error_response(exc)
        except HTTPException as exc:
            self.log("调用失败", status="failed", error=str(exc.detail))
            raise
        except Exception as exc:
            self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""))
            if self.endpoint.startswith("/v1/images"):
                return _image_error_response(exc)
            return _protocol_error_response(exc, 502, sse)
        if not has_first:
            self.log("流式调用结束")
            return StreamingResponse(sender(()), media_type="text/event-stream")
        return StreamingResponse(sender(self.stream(itertools.chain([first], result))), media_type="text/event-stream")

    def stream(self, items):
        urls: list[str] = []
        account_emails: list[str] = []
        conversation_ids: list[str] = []
        failed = False
        try:
            for item in items:
                urls.extend(_collect_urls(item))
                account_emails.extend(_collect_account_emails(item))
                conversation_ids.extend(_collect_conversation_ids(item))
                yield _strip_internal_response_fields(item)
        except Exception as exc:
            failed = True
            self.log(
                "流式调用失败",
                status="failed",
                error=str(exc),
                urls=urls,
                account_email=(account_emails[0] if account_emails else getattr(exc, "account_email", "")),
                conversation_id=(conversation_ids[0] if conversation_ids else getattr(exc, "conversation_id", "")),
            )
            if self.endpoint.startswith("/v1/images") and not hasattr(exc, "to_openai_error"):
                from services.protocol.conversation import ImageGenerationError, public_image_error_message

                raise ImageGenerationError(public_image_error_message(str(exc))) from exc
            raise
        finally:
            if not failed:
                self.log("流式调用结束", urls=urls, account_email=account_emails[0] if account_emails else "",
                         conversation_id=conversation_ids[0] if conversation_ids else "")

    def log(self, suffix: str, result: object = None, status: str = "success", error: str = "",
            urls: list[str] | None = None, account_email: str = "", conversation_id: str = "") -> None:
        detail = {
            "key_id": self.identity.get("id"),
            "key_name": self.identity.get("name"),
            "role": self.identity.get("role"),
            "endpoint": self.endpoint,
            "model": self.model,
            "started_at": datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_ms": int((time.time() - self.started) * 1000),
            "status": status,
        }
        request_excerpt = _request_excerpt(self.request_text)
        if request_excerpt:
            detail["request_text"] = request_excerpt
        if self.request_shape:
            detail["request_shape"] = self.request_shape
        if error:
            detail["error"] = error
        email = str(account_email or "").strip()
        if not email:
            emails = _collect_account_emails(result)
            email = emails[0] if emails else ""
        if email:
            detail["account_email"] = email
        conv_id = str(conversation_id or "").strip()
        if not conv_id:
            conv_ids = _collect_conversation_ids(result)
            conv_id = conv_ids[0] if conv_ids else ""
        if conv_id:
            detail["conversation_id"] = conv_id
        collected_urls = [*(urls or []), *_collect_urls(result)]
        if collected_urls and not self.endpoint.startswith("/v1/search"):
            detail["urls"] = list(dict.fromkeys(collected_urls))
        log_service.add(LOG_TYPE_CALL, f"{self.summary}{suffix}", detail)
