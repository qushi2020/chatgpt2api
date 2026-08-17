from __future__ import annotations

import unittest
from unittest.mock import patch

from services import cpa_service, sub2api_service


class Sub2ApiCredentialExtractionTest(unittest.TestCase):
    """sub2api 导入必须保留 refresh_token，否则账号无法自动刷新而失效。"""

    def test_extract_credentials_keeps_refresh_token(self):
        creds = {
            "access_token": "at-123",
            "refresh_token": "rt-456",
            "email": "user@example.com",
        }
        result = sub2api_service._extract_account_credentials(creds)
        self.assertEqual(result["access_token"], "at-123")
        self.assertEqual(result["refresh_token"], "rt-456")
        self.assertEqual(result["email"], "user@example.com")

    def test_extract_credentials_missing_refresh_token_is_empty_not_error(self):
        # 旧版/部分源不返回 refresh_token 时不应报错，只是留空
        result = sub2api_service._extract_account_credentials({"access_token": "at-only"})
        self.assertEqual(result["access_token"], "at-only")
        self.assertEqual(result["refresh_token"], "")

    def test_fetch_account_credentials_builds_payloads_with_refresh_token(self):
        server = {"base_url": "https://api.example.test", "api_key": "k", "id": "s1"}

        class Resp:
            ok = True

            def json(self):
                return {"code": 0, "data": {"accounts": [
                    {"id": "a1", "name": "u1@example.com",
                     "credentials": {"access_token": "at-1", "refresh_token": "rt-1", "email": "u1@example.com"}},
                    {"id": "a2", "name": "u2@example.com",
                     "credentials": {"access_token": "at-2", "email": "u2@example.com"}},
                ]}}

        class Sess:
            def get(self, *a, **k):
                return Resp()

            def close(self):
                pass

        with patch.object(sub2api_service, "Session", return_value=Sess()):
            payloads, errors = sub2api_service._fetch_account_credentials(server, ["a1", "a2"])

        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["refresh_token"], "rt-1")
        self.assertEqual(payloads[0]["source_type"], "codex")
        self.assertEqual(payloads[1]["refresh_token"], "")
        self.assertEqual(errors, [])

    def test_run_import_stores_refresh_token_via_add_account_items(self):
        captured = {}

        def fake_fetch(server, ids):
            return [{"access_token": "at-x", "refresh_token": "rt-x",
                     "email": "x@example.com", "source_type": "codex"}], []

        def fake_add_items(payloads):
            captured["payloads"] = payloads
            return {"added": 1, "skipped": 0, "items": []}

        cfg = sub2api_service.Sub2APIConfig(sub2api_service.SUB2API_CONFIG_FILE)
        cfg._servers = [{"id": "s1", "base_url": "https://api.example.test", "api_key": "k"}]
        svc = sub2api_service.Sub2APIImportService(cfg)

        with patch.object(sub2api_service, "_fetch_account_credentials", fake_fetch), \
             patch.object(sub2api_service.account_service, "add_account_items", fake_add_items), \
             patch.object(sub2api_service.account_service, "refresh_accounts", lambda toks: {"refreshed": 0}):
            svc._run_import("s1", {"id": "s1"}, ["a1"])

        self.assertEqual(captured["payloads"][0]["refresh_token"], "rt-x")


class CpaCredentialExtractionTest(unittest.TestCase):
    """CPA 导入同样必须保留 refresh_token。"""

    def test_fetch_remote_access_token_returns_refresh_token(self):
        pool = {"base_url": "https://cpa.example.test", "secret_key": "sk"}

        class Resp:
            ok = True

            def json(self):
                return {"access_token": "at-cpa", "refresh_token": "rt-cpa", "email": "c@example.com"}

        class Sess:
            def get(self, *a, **k):
                return Resp()

            def close(self):
                pass

        with patch.object(cpa_service, "Session", return_value=Sess()):
            payload, error = cpa_service.fetch_remote_access_token(pool, "file.json")

        self.assertIsNone(error)
        self.assertEqual(payload["access_token"], "at-cpa")
        self.assertEqual(payload["refresh_token"], "rt-cpa")
        self.assertEqual(payload["source_type"], "codex")


if __name__ == "__main__":
    unittest.main()
