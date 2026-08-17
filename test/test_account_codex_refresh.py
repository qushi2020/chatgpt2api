from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.account_service import AccountService
from services.config import config
from services.openai_backend_api import InvalidAccessTokenError, OpenAIBackendAPI
from services.storage.json_storage import JSONStorageBackend


def make_jwt(payload: dict) -> str:
    def encode(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f'{encode({"alg": "none", "typ": "JWT"})}.{encode(payload)}.sig'


class FakeRefreshResponse:
    status_code = 200
    text = "{}"

    @staticmethod
    def json() -> dict[str, str]:
        return {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "id_token": "new-id-token",
        }


class FakeRefreshSession:
    def __init__(self) -> None:
        self.request_kwargs: dict | None = None
        self.closed = False

    def post(self, url: str, **kwargs):
        self.request_kwargs = {"url": url, **kwargs}
        return FakeRefreshResponse()

    def close(self) -> None:
        self.closed = True


class FakeInvalidGrantResponse:
    status_code = 400
    text = "invalid_grant: refresh token expired"

    @staticmethod
    def json() -> dict:
        raise ValueError("not json")


class FakeInvalidGrantSession(FakeRefreshSession):
    def post(self, url: str, **kwargs):
        self.request_kwargs = {"url": url, **kwargs}
        return FakeInvalidGrantResponse()


class FakeUsageResponse:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class FakeUsageSession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.request_kwargs: dict | None = None

    def get(self, url: str, **kwargs):
        self.request_kwargs = {"url": url, **kwargs}
        return FakeUsageResponse(self.payload)

    def close(self) -> None:
        return None


class CodexAccountRefreshTests(unittest.TestCase):
    def test_codex_availability_normalizes_string_booleans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))

            unavailable = service._normalize_account(
                {"access_token": "token-false", "source_type": "codex", "codex_available": "false"}
            )
            available = service._normalize_account(
                {"access_token": "token-true", "source_type": "codex", "codex_available": "true"}
            )

        self.assertIsNotNone(unavailable)
        self.assertIsNotNone(available)
        self.assertFalse(unavailable["codex_available"])
        self.assertTrue(available["codex_available"])

    def test_codex_usage_helpers_ignore_non_finite_and_out_of_range_values(self) -> None:
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.assertIsNone(OpenAIBackendAPI._codex_window_remaining({"used_percent": value}))
                self.assertIsNone(OpenAIBackendAPI._codex_remaining_percent(value))
                self.assertIsNone(OpenAIBackendAPI._codex_reset_iso({"reset_at": value}))

        self.assertIsNone(OpenAIBackendAPI._codex_reset_iso({"reset_at": 10**30}))
        self.assertIsNone(OpenAIBackendAPI._codex_reset_iso({}))

    def test_codex_oauth_client_uses_token_claim_then_official_fallback(self) -> None:
        token = make_jwt({"client_id": "app_from_token"})

        self.assertEqual(
            AccountService._oauth_client_id_for_account(
                {"access_token": token, "source_type": "codex"}
            ),
            "app_from_token",
        )
        self.assertEqual(
            AccountService._oauth_client_id_for_account(
                {"access_token": "opaque-token", "source_type": "codex"}
            ),
            AccountService._CODEX_OAUTH_CLIENT_ID,
        )
        self.assertEqual(
            AccountService._oauth_client_id_for_account(
                {"access_token": "opaque-token", "source_type": "web"}
            ),
            AccountService._OAUTH_CLIENT_ID,
        )

    def test_codex_refresh_uses_json_and_source_specific_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            session = FakeRefreshSession()

            with patch("curl_cffi.requests.Session", return_value=session):
                tokens = service._request_access_token_refresh(
                    "refresh-token",
                    {"access_token": "opaque-token", "source_type": "codex"},
                )

        self.assertEqual(tokens["access_token"], "new-access-token")
        self.assertTrue(session.closed)
        self.assertIsNotNone(session.request_kwargs)
        self.assertNotIn("data", session.request_kwargs)
        self.assertEqual(
            session.request_kwargs["json"],
            {
                "grant_type": "refresh_token",
                "refresh_token": "refresh-token",
                "client_id": AccountService._CODEX_OAUTH_CLIENT_ID,
            },
        )
        self.assertEqual(session.request_kwargs["headers"]["Content-Type"], "application/json")

    def test_non_json_invalid_grant_is_classified_as_permanent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            session = FakeInvalidGrantSession()

            with (
                patch("curl_cffi.requests.Session", return_value=session),
                self.assertRaisesRegex(RuntimeError, "invalid_grant") as raised,
            ):
                service._request_access_token_refresh(
                    "refresh-token",
                    {"access_token": "opaque-token", "source_type": "codex"},
                )

        self.assertEqual(service._classify_token_refresh_error(str(raised.exception)), "permanent")
        self.assertTrue(session.closed)

    def test_invalid_refresh_token_is_classified_as_permanent(self) -> None:
        self.assertEqual(
            AccountService._classify_token_refresh_error(
                "oauth_refresh_http_401: code=invalid_refresh_token"
            ),
            "permanent",
        )

    def test_web_refresh_keeps_form_encoded_platform_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            session = FakeRefreshSession()

            with patch("curl_cffi.requests.Session", return_value=session):
                service._request_access_token_refresh(
                    "refresh-token",
                    {"access_token": "opaque-token", "source_type": "web"},
                )

        self.assertIsNotNone(session.request_kwargs)
        self.assertNotIn("json", session.request_kwargs)
        self.assertEqual(
            session.request_kwargs["data"],
            {
                "grant_type": "refresh_token",
                "refresh_token": "refresh-token",
                "client_id": AccountService._OAUTH_CLIENT_ID,
            },
        )
        self.assertEqual(
            session.request_kwargs["headers"]["Content-Type"],
            "application/x-www-form-urlencoded",
        )

    def test_codex_refresh_uses_usage_check_and_restores_false_abnormal(self) -> None:
        original_auto_remove = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items(
                    [
                        {
                            "type": "codex",
                            "access_token": "codex-token",
                            "status": "异常",
                            "quota": 0,
                        }
                    ]
                )

                with (
                    patch.object(
                        OpenAIBackendAPI,
                        "get_codex_user_info",
                        return_value={
                            "type": "plus",
                            "status": "正常",
                            "codex_available": True,
                            "codex_remaining_percent": 42,
                            "image_quota_unknown": True,
                        },
                    ) as codex_check,
                    patch.object(
                        OpenAIBackendAPI,
                        "get_user_info",
                        side_effect=AssertionError("web account check must not run for Codex accounts"),
                    ),
                ):
                    result = service.refresh_accounts(
                        ["codex-token"],
                        defer_invalid_removal=False,
                    )

                account = service.get_account("codex-token")

            self.assertEqual(result["refreshed"], 1)
            self.assertEqual(result["errors"], [])
            self.assertIsNotNone(account)
            self.assertEqual(account["status"], "正常")
            self.assertEqual(account["quota"], 0)
            self.assertTrue(account["codex_available"])
            self.assertEqual(account["codex_remaining_percent"], 42)
            self.assertTrue(account["image_quota_unknown"])
            self.assertEqual(account["type"], "plus")
            codex_check.assert_called_once_with()
        finally:
            if original_auto_remove is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_auto_remove

    def test_codex_usage_is_parsed_as_scheduler_availability(self) -> None:
        token = make_jwt(
            {
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "acct_123",
                    "chatgpt_plan_type": "plus",
                },
                "https://api.openai.com/profile": {"email": "codex@example.com"},
            }
        )
        backend = OpenAIBackendAPI(token)
        backend.account = {"source_type": "codex", "type": "free"}
        try:
            with patch.object(
                backend,
                "_get_codex_usage",
                return_value={
                    "plan_type": "pro",
                    "rate_limit": {
                        "allowed": True,
                        "limit_reached": False,
                        "primary_window": {"used_percent": 25, "reset_at": 1_800_000_000},
                        "secondary_window": {"used_percent": 40, "reset_at": 1_800_100_000},
                    },
                },
            ):
                result = backend.get_codex_user_info()
        finally:
            backend.close()

        self.assertEqual(result["email"], "codex@example.com")
        self.assertEqual(result["account_id"], "acct_123")
        self.assertEqual(result["type"], "pro")
        self.assertEqual(result["status"], "正常")
        self.assertTrue(result["codex_available"])
        self.assertEqual(result["codex_remaining_percent"], 60)
        self.assertTrue(result["image_quota_unknown"])
        self.assertNotIn("quota", result)
        self.assertEqual(len(result["limits_progress"]), 2)
        self.assertEqual(result["restore_at"], "2027-01-16T11:46:40+00:00")

    def test_codex_limited_restore_uses_only_exhausted_window(self) -> None:
        backend = OpenAIBackendAPI("opaque-token")
        backend.account = {"source_type": "codex", "type": "plus"}
        try:
            with patch.object(
                backend,
                "_get_codex_usage",
                return_value={
                    "rate_limit": {
                        "allowed": False,
                        "limit_reached": True,
                        "primary_window": {"used_percent": 100, "reset_at": 1_800_000_000},
                        "secondary_window": {"used_percent": 20, "reset_at": 1_800_100_000},
                    },
                },
            ):
                result = backend.get_codex_user_info()
        finally:
            backend.close()

        self.assertEqual(result["restore_at"], "2027-01-15T08:00:00+00:00")

    def test_codex_usage_marks_only_source_limit_as_limited(self) -> None:
        backend = OpenAIBackendAPI("opaque-token")
        backend.account = {"source_type": "codex", "type": "plus"}
        try:
            with patch.object(
                backend,
                "_get_codex_usage",
                return_value={
                    "plan_type": "plus",
                    "rate_limit": {
                        "allowed": False,
                        "limit_reached": True,
                        "primary_window": {"used_percent": 100, "reset_at": 1_800_000_000},
                        "secondary_window": {"used_percent": 100, "reset_at": 1_800_100_000},
                    },
                },
            ):
                result = backend.get_codex_user_info()
        finally:
            backend.close()

        self.assertEqual(result["status"], "限流")
        self.assertFalse(result["codex_available"])
        self.assertEqual(result["codex_remaining_percent"], 0)
        self.assertNotIn("quota", result)
        self.assertEqual(result["restore_at"], "2027-01-16T11:46:40+00:00")

    def test_codex_usage_uses_current_wham_route(self) -> None:
        backend = OpenAIBackendAPI("opaque-token")
        backend.session.close()
        session = FakeUsageSession({"plan_type": "plus"})
        backend.session = session
        try:
            result = backend._get_codex_usage()
        finally:
            backend.close()

        self.assertEqual(result, {"plan_type": "plus"})
        self.assertIsNotNone(session.request_kwargs)
        self.assertEqual(session.request_kwargs["url"], "https://chatgpt.com/backend-api/wham/usage")

    def test_codex_usage_honors_global_blocking_signals(self) -> None:
        payloads = {
            "spend_control": {
                "rate_limit": {"allowed": True, "limit_reached": False},
                "spend_control": {"reached": True},
            },
            "rate_limit_reached_type": {
                "rate_limit": {"allowed": True, "limit_reached": False},
                "rate_limit_reached_type": "rate_limit_reached",
            },
        }

        for signal, payload in payloads.items():
            with self.subTest(signal=signal):
                backend = OpenAIBackendAPI("opaque-token")
                backend.account = {"source_type": "codex", "type": "plus"}
                try:
                    with patch.object(backend, "_get_codex_usage", return_value=payload):
                        result = backend.get_codex_user_info()
                finally:
                    backend.close()

                self.assertEqual(result["status"], "限流")
                self.assertFalse(result["codex_available"])

    def test_unknown_rate_limit_reached_type_does_not_block_account(self) -> None:
        backend = OpenAIBackendAPI("opaque-token")
        backend.account = {"source_type": "codex", "type": "plus"}
        try:
            with patch.object(
                backend,
                "_get_codex_usage",
                return_value={
                    "rate_limit": {
                        "allowed": True,
                        "limit_reached": False,
                        "primary_window": {"used_percent": 20, "reset_at": 1_800_000_000},
                    },
                    "rate_limit_reached_type": "future_non_blocking_state",
                },
            ):
                result = backend.get_codex_user_info()
        finally:
            backend.close()

        self.assertEqual(result["status"], "正常")
        self.assertTrue(result["codex_available"])
        self.assertIsNone(result["codex_rate_limit_reached_type"])

    def test_additional_rate_limit_does_not_block_whole_account(self) -> None:
        backend = OpenAIBackendAPI("opaque-token")
        backend.account = {"source_type": "codex", "type": "plus"}
        try:
            with patch.object(
                backend,
                "_get_codex_usage",
                return_value={
                    "rate_limit": {
                        "allowed": True,
                        "limit_reached": False,
                        "primary_window": {"used_percent": 20, "reset_at": 1_800_000_000},
                    },
                    "additional_rate_limits": [
                        {
                            "metered_feature": "gpt-5.3-codex-spark",
                            "limit_name": "Spark",
                            "rate_limit": {
                                "allowed": False,
                                "limit_reached": True,
                                "primary_window": {"used_percent": 100, "reset_at": 1_800_100_000},
                            },
                        }
                    ],
                },
            ):
                result = backend.get_codex_user_info()
        finally:
            backend.close()

        self.assertEqual(result["status"], "正常")
        self.assertTrue(result["codex_available"])
        self.assertEqual(result["codex_remaining_percent"], 80)
        self.assertEqual(len(result["limits_progress"]), 2)

    def test_spend_control_individual_limit_supplies_remaining_and_restore_time(self) -> None:
        backend = OpenAIBackendAPI("opaque-token")
        backend.account = {"source_type": "codex", "type": "plus"}
        try:
            with patch.object(
                backend,
                "_get_codex_usage",
                return_value={
                    "spend_control": {
                        "reached": True,
                        "individual_limit": {
                            "limit": "100",
                            "used": "100",
                            "remaining_percent": 0,
                            "reset_at": 1_800_200_000,
                        },
                    },
                },
            ):
                result = backend.get_codex_user_info()
        finally:
            backend.close()

        self.assertEqual(result["status"], "限流")
        self.assertFalse(result["codex_available"])
        self.assertEqual(result["codex_remaining_percent"], 0)
        self.assertEqual(result["restore_at"], "2027-01-17T15:33:20+00:00")
        self.assertEqual(result["limits_progress"][0]["feature_name"], "codex_spend_control")

    def test_truly_invalid_codex_token_is_still_removed(self) -> None:
        original_auto_remove = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items(
                    [
                        {
                            "type": "codex",
                            "access_token": "invalid-codex-token",
                            "status": "正常",
                            "quota": 50,
                        }
                    ]
                )

                with patch.object(
                    OpenAIBackendAPI,
                    "get_codex_user_info",
                    side_effect=InvalidAccessTokenError(
                        "token invalidated (/backend-api/wham/usage)"
                    ),
                ):
                    result = service.refresh_accounts(
                        ["invalid-codex-token"],
                        defer_invalid_removal=False,
                    )

                account = service.get_account("invalid-codex-token")

            self.assertEqual(result["refreshed"], 0)
            self.assertEqual(len(result["errors"]), 1)
            self.assertIsNone(account)
        finally:
            if original_auto_remove is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_auto_remove

    def test_invalid_oauth_client_does_not_mark_codex_account_invalid(self) -> None:
        original_auto_remove = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items(
                    [
                        {
                            "type": "codex",
                            "access_token": "codex-token",
                            "refresh_token": "refresh-token",
                            "status": "正常",
                            "quota": 50,
                        }
                    ]
                )

                with (
                    patch.object(
                        OpenAIBackendAPI,
                        "get_codex_user_info",
                        side_effect=InvalidAccessTokenError(
                            "token invalidated (/backend-api/wham/usage)"
                        ),
                    ),
                    patch.object(
                        service,
                        "_request_access_token_refresh",
                        side_effect=RuntimeError(
                            "oauth_refresh_http_401: invalid_client: Invalid client specified."
                        ),
                    ),
                ):
                    result = service.refresh_accounts(
                        ["codex-token"],
                        defer_invalid_removal=False,
                    )

                account = service.get_account("codex-token")

            self.assertEqual(result["refreshed"], 0)
            self.assertEqual(len(result["errors"]), 1)
            self.assertIn("OAuth client mismatch", result["errors"][0]["error"])
            self.assertIsNotNone(account)
            self.assertEqual(account["status"], "正常")
            self.assertEqual(account["quota"], 50)
            self.assertEqual(account["invalid_count"], 0)
        finally:
            if original_auto_remove is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_auto_remove

    def test_transient_refresh_failure_preserves_codex_account(self) -> None:
        original_auto_remove = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items(
                    [
                        {
                            "type": "codex",
                            "access_token": "codex-token",
                            "refresh_token": "refresh-token",
                            "status": "正常",
                            "quota": 50,
                        }
                    ]
                )

                with (
                    patch.object(
                        OpenAIBackendAPI,
                        "get_codex_user_info",
                        side_effect=InvalidAccessTokenError(
                            "token invalidated (/backend-api/wham/usage)"
                        ),
                    ),
                    patch.object(
                        service,
                        "_request_access_token_refresh",
                        side_effect=TimeoutError("OAuth request timed out"),
                    ),
                ):
                    result = service.refresh_accounts(
                        ["codex-token"],
                        defer_invalid_removal=False,
                    )

                account = service.get_account("codex-token")

            self.assertEqual(result["refreshed"], 0)
            self.assertEqual(len(result["errors"]), 1)
            self.assertIn("OAuth transient error", result["errors"][0]["error"])
            self.assertIsNotNone(account)
            self.assertEqual(account["status"], "正常")
            self.assertEqual(account["invalid_count"], 0)
            self.assertEqual(account["last_token_refresh_error_kind"], "transient")
        finally:
            if original_auto_remove is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_auto_remove

    def test_recent_transient_refresh_backoff_preserves_codex_account(self) -> None:
        original_auto_remove = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                access_token = make_jwt({"exp": 1})
                service.add_account_items(
                    [
                        {
                            "type": "codex",
                            "access_token": access_token,
                            "refresh_token": "refresh-token",
                            "status": "正常",
                        }
                    ]
                )
                service._record_token_refresh_error(
                    access_token,
                    "test",
                    "OAuth request timed out",
                    "transient",
                )

                with (
                    patch.object(
                        OpenAIBackendAPI,
                        "get_codex_user_info",
                        side_effect=InvalidAccessTokenError(
                            "token invalidated (/backend-api/wham/usage)"
                        ),
                    ),
                    patch.object(
                        service,
                        "_request_access_token_refresh",
                        side_effect=AssertionError("refresh backoff must be respected"),
                    ),
                ):
                    result = service.refresh_accounts(
                        [access_token],
                        defer_invalid_removal=False,
                    )

                account = service.get_account(access_token)

            self.assertEqual(result["refreshed"], 0)
            self.assertEqual(len(result["errors"]), 1)
            self.assertIn("OAuth transient error", result["errors"][0]["error"])
            self.assertIsNotNone(account)
            self.assertEqual(account["invalid_count"], 0)
        finally:
            if original_auto_remove is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_auto_remove

    def test_permanent_refresh_failure_can_remove_invalid_codex_account(self) -> None:
        original_auto_remove = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items(
                    [
                        {
                            "type": "codex",
                            "access_token": "codex-token",
                            "refresh_token": "refresh-token",
                            "status": "正常",
                        }
                    ]
                )

                with (
                    patch.object(
                        OpenAIBackendAPI,
                        "get_codex_user_info",
                        side_effect=InvalidAccessTokenError(
                            "token invalidated (/backend-api/wham/usage)"
                        ),
                    ),
                    patch.object(
                        service,
                        "_request_access_token_refresh",
                        side_effect=RuntimeError("oauth_refresh_http_400: invalid_grant"),
                    ),
                ):
                    result = service.refresh_accounts(
                        ["codex-token"],
                        defer_invalid_removal=False,
                    )

            self.assertEqual(result["refreshed"], 0)
            self.assertEqual(len(result["errors"]), 1)
            self.assertIsNone(service.get_account("codex-token"))
        finally:
            if original_auto_remove is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_auto_remove

    def test_app_session_terminated_starts_relogin_without_removing_account(self) -> None:
        original_auto_remove = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items(
                    [
                        {
                            "type": "codex",
                            "access_token": "codex-token",
                            "refresh_token": "refresh-token",
                            "email": "codex@example.com",
                            "password": "secret",
                            "status": "正常",
                            "codex_available": True,
                            "codex_remaining_percent": 75,
                        }
                    ]
                )

                with (
                    patch.object(
                        OpenAIBackendAPI,
                        "get_codex_user_info",
                        side_effect=InvalidAccessTokenError(
                            "token invalidated (/backend-api/wham/usage)"
                        ),
                    ),
                    patch.object(
                        service,
                        "_request_access_token_refresh",
                        side_effect=RuntimeError("oauth_refresh_http_401: app_session_terminated"),
                    ),
                    patch("services.account_service.Thread") as thread_class,
                ):
                    result = service.refresh_accounts(
                        ["codex-token"],
                        defer_invalid_removal=False,
                    )

                account = service.get_account("codex-token")

            self.assertEqual(result["refreshed"], 0)
            self.assertEqual(len(result["errors"]), 1)
            self.assertIn("OAuth reauth required error", result["errors"][0]["error"])
            self.assertIsNotNone(account)
            self.assertEqual(account["status"], "正常")
            self.assertEqual(account["invalid_count"], 0)
            self.assertEqual(account["last_token_refresh_error_kind"], "reauth_required")
            self.assertFalse(service._is_image_account_available(account))
            self.assertEqual(service.get_text_access_token(), "")
            self.assertEqual(service.get_stats()["active"], 0)
            self.assertEqual(service.get_stats()["abnormal"], 1)
            self.assertEqual(service.get_stats()["reauth_required"], 1)
            thread_class.assert_called_once()
            thread_class.return_value.start.assert_called_once_with()
        finally:
            if original_auto_remove is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_auto_remove

    def test_expired_token_starts_only_one_password_relogin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            access_token = make_jwt({"exp": 1})
            service.add_account_items(
                [
                    {
                        "type": "codex",
                        "access_token": access_token,
                        "refresh_token": "refresh-token",
                        "email": "codex@example.com",
                        "password": "secret",
                        "status": "正常",
                    }
                ]
            )

            with (
                patch.object(
                    OpenAIBackendAPI,
                    "get_codex_user_info",
                    side_effect=InvalidAccessTokenError(
                        "token invalidated (/backend-api/wham/usage)"
                    ),
                ),
                patch.object(
                    service,
                    "_request_access_token_refresh",
                    side_effect=RuntimeError("oauth_refresh_http_401: app_session_terminated"),
                ),
                patch("services.account_service.Thread") as thread_class,
            ):
                result = service.refresh_accounts(
                    [access_token],
                    defer_invalid_removal=False,
                )

        self.assertEqual(result["refreshed"], 0)
        self.assertEqual(thread_class.call_count, 1)
        thread_class.return_value.start.assert_called_once_with()

    def test_password_relogin_dedupe_follows_rotated_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [
                    {
                        "access_token": "old-token",
                        "refresh_token": "old-refresh-token",
                        "email": "codex@example.com",
                        "password": "secret",
                    }
                ]
            )

            with patch("services.account_service.Thread"):
                self.assertTrue(
                    service._start_password_relogin(
                        "old-token",
                        "codex@example.com",
                        "secret",
                        "test",
                    )
                )
                service._apply_refreshed_tokens(
                    "old-token",
                    {
                        "access_token": "new-token",
                        "refresh_token": "new-refresh-token",
                    },
                    "test",
                )
                self.assertFalse(
                    service._start_password_relogin(
                        "old-token",
                        "codex@example.com",
                        "secret",
                        "test",
                    )
                )

            self.assertNotIn("old-token", service._relogin_inflight)
            self.assertIn("new-token", service._relogin_inflight)

    def test_successful_password_relogin_preserves_codex_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [
                    {
                        "type": "codex",
                        "access_token": "old-token",
                        "status": "正常",
                        "quota": 99,
                        "codex_available": True,
                        "codex_remaining_percent": 75,
                    }
                ]
            )

            with patch.object(
                service,
                "_login_with_password",
                return_value={
                    "ok": True,
                    "access_token": "new-token",
                    "refresh_token": "new-refresh-token",
                    "id_token": "new-id-token",
                    "source_type": "password",
                },
            ):
                service._password_re_login_thread(
                    "old-token",
                    "codex@example.com",
                    "secret",
                    "test",
                )

            account = service.get_account("new-token")

        self.assertIsNotNone(account)
        self.assertEqual(account["source_type"], "codex")
        self.assertTrue(account["image_quota_unknown"])
        self.assertIsNone(account["codex_available"])
        self.assertIsNone(account["codex_remaining_percent"])
        self.assertEqual(account["quota"], 0)
        self.assertEqual(account["status"], "正常")

    def test_password_relogin_network_error_preserves_account(self) -> None:
        original_auto_remove = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items(
                    [
                        {
                            "type": "codex",
                            "access_token": "codex-token",
                            "status": "正常",
                            "quota": 17,
                            "codex_available": True,
                        }
                    ]
                )

                with patch.object(
                    service,
                    "_login_with_password",
                    side_effect=TimeoutError("login request timed out"),
                ):
                    service._password_re_login_thread(
                        "codex-token",
                        "codex@example.com",
                        "secret",
                        "test",
                    )

                account = service.get_account("codex-token")

            self.assertIsNotNone(account)
            self.assertEqual(account["status"], "正常")
            self.assertEqual(account["quota"], 17)
            self.assertTrue(account["codex_available"])
            self.assertEqual(account["source_type"], "codex")
            self.assertEqual(account["last_relogin_error"], "login request timed out")
        finally:
            if original_auto_remove is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_auto_remove

    def test_codex_quota_is_excluded_from_image_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [
                    {"type": "codex", "access_token": "codex-token", "status": "正常", "quota": 99},
                    {"access_token": "web-token", "status": "正常", "quota": 7},
                ]
            )
            service.init_refresh_progress("refresh", 1)
            service.update_refresh_progress("refresh", "codex-token")

            self.assertEqual(service.get_stats()["total_quota"], 7)
            self.assertEqual(service.get_stats()["unknown_quota_count"], 1)
            self.assertEqual(service.get_refresh_progress("refresh")["total_quota"], 0)
            self.assertEqual(service.get_refresh_progress("refresh")["unknown_quota_count"], 1)

    def test_codex_success_keeps_remote_usage_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [
                    {
                        "type": "codex",
                        "access_token": "codex-token",
                        "status": "正常",
                        "quota": 35,
                    }
                ]
            )

            updated = service.mark_image_result("codex-token", success=True)

        self.assertIsNotNone(updated)
        self.assertEqual(updated["quota"], 35)
        self.assertEqual(updated["status"], "正常")
        self.assertEqual(updated["success"], 1)


if __name__ == "__main__":
    unittest.main()
