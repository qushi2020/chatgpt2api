from __future__ import annotations

import copy
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.image_quota as image_quota_module


AUTH_HEADERS = {"Authorization": "Bearer test-key"}


class FakeAccountService:
    def __init__(self) -> None:
        self.accounts = [
            {
                "access_token": "token-a",
                "status": "正常",
                "quota": 3,
                "image_quota_unknown": False,
                "success": 2,
                "fail": 1,
            },
            {
                "access_token": "token-b",
                "status": "正常",
                "quota": 0,
                "image_quota_unknown": True,
                "success": 0,
                "fail": 0,
            },
            {
                "access_token": "token-c",
                "status": "限流",
                "quota": 5,
                "image_quota_unknown": False,
                "success": 1,
                "fail": 0,
            },
        ]
        self.refresh_calls: list[list[str]] = []

    def list_accounts(self) -> list[dict]:
        return copy.deepcopy(self.accounts)

    def refresh_accounts(self, access_tokens: list[str]) -> dict:
        self.refresh_calls.append(list(access_tokens))
        self.accounts[0]["quota"] = 8
        self.accounts[2]["status"] = "正常"
        self.accounts[2]["quota"] = 2
        return {"refreshed": len(access_tokens), "errors": [], "items": self.list_accounts(), "relogined": 0}


class ImageQuotaApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_account_service = FakeAccountService()
        self.patchers = [
            mock.patch.object(image_quota_module, "account_service", self.fake_account_service),
            mock.patch.object(image_quota_module, "require_identity", lambda _authorization: {"id": "test"}),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        with image_quota_module._refresh_lock:
            image_quota_module._refresh_state.update(
                {
                    "running": False,
                    "lastStartedAt": None,
                    "lastFinishedAt": None,
                    "attemptedCount": 0,
                    "successCount": 0,
                    "failureCount": 0,
                    "lastError": None,
                }
            )
        app = FastAPI()
        app.include_router(image_quota_module.create_router())
        self.client = TestClient(app)

    def test_codex_availability_uses_codex_health_signal(self) -> None:
        account = {
            "source_type": "codex",
            "status": "正常",
            "quota": 0,
            "image_quota_unknown": True,
        }

        self.assertFalse(image_quota_module._is_image_available(account))
        self.assertFalse(image_quota_module._is_image_available({**account, "codex_available": False}))
        self.assertTrue(image_quota_module._is_image_available({**account, "codex_available": True}))
        self.assertFalse(
            image_quota_module._is_image_available(
                {
                    **account,
                    "codex_available": True,
                    "last_token_refresh_error_kind": "reauth_required",
                }
            )
        )

    def test_disabled_refresh_returns_codexmanager_compatible_payload(self) -> None:
        response = self.client.get("/image-quota?refresh=0", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["object"], "image_quota")
        self.assertEqual(payload["totalRemaining"], 3)
        self.assertEqual(payload["remaining"], 3)
        self.assertTrue(payload["totalRemainingKnown"])
        self.assertEqual(payload["accountCount"], 3)
        self.assertEqual(payload["availableCount"], 2)
        self.assertEqual(payload["unknownQuotaCount"], 1)
        self.assertEqual(payload["limitedCount"], 1)
        self.assertEqual(payload["refresh"]["mode"], "disabled")
        self.assertEqual(self.fake_account_service.refresh_calls, [])

    def test_sync_refresh_updates_accounts_before_returning_payload(self) -> None:
        response = self.client.get("/v1/image-quota?refresh=sync", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(self.fake_account_service.refresh_calls, [["token-a", "token-b", "token-c"]])
        self.assertEqual(payload["totalRemaining"], 10)
        self.assertEqual(payload["availableCount"], 3)
        self.assertEqual(payload["refresh"]["mode"], "sync")
        self.assertEqual(payload["refresh"]["attemptedCount"], 3)
        self.assertEqual(payload["refresh"]["successCount"], 3)
        self.assertEqual(payload["refresh"]["failureCount"], 0)
        self.assertIsInstance(payload["lastRefreshedAt"], int)

    def test_quota_route_aliases_are_available(self) -> None:
        for path in ["/v1/image-quota", "/images/quota", "/v1/images/quota"]:
            with self.subTest(path=path):
                response = self.client.get(f"{path}?refresh=0", headers=AUTH_HEADERS)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["object"], "image_quota")

    def test_x_api_key_header_is_accepted(self) -> None:
        seen: list[str | None] = []

        def fake_require_identity(authorization: str | None) -> dict[str, str]:
            seen.append(authorization)
            return {"id": "test"}

        with mock.patch.object(image_quota_module, "require_identity", fake_require_identity):
            response = self.client.get("/image-quota?refresh=0", headers={"X-API-Key": "key-from-header"})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(seen, ["Bearer key-from-header"])


if __name__ == "__main__":
    unittest.main()
