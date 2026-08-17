from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from services.openai_backend_api import OpenAIBackendAPI
from services.protocol import web_search_tool


class BackendResourceLifecycleTests(unittest.TestCase):
    def test_get_user_info_does_not_share_session_across_worker_threads(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "test-token"

        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_request(payload: dict[str, object]):
            def run() -> dict[str, object]:
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                with lock:
                    active -= 1
                return payload

            return run

        with (
            patch.object(backend, "_get_me", side_effect=fake_request({"email": "test@example.com", "id": "user-1"})),
            patch.object(backend, "_get_conversation_init", side_effect=fake_request({"limits_progress": [], "default_model_slug": "auto"})),
            patch.object(backend, "_get_default_account", side_effect=fake_request({"plan_type": "free"})),
        ):
            result = backend.get_user_info()

        self.assertEqual(result["email"], "test@example.com")
        self.assertEqual(max_active, 1)

    def test_web_search_closes_backend_on_success(self) -> None:
        class FakeBackend:
            def __init__(self, _token: str) -> None:
                self.closed = False

            def search(self, _query: str) -> dict[str, object]:
                return {"answer": "ok"}

            def close(self) -> None:
                self.closed = True

        fake_backend: FakeBackend | None = None

        def build_backend(token: str) -> FakeBackend:
            nonlocal fake_backend
            fake_backend = FakeBackend(token)
            return fake_backend

        with (
            patch.object(web_search_tool.account_service, "get_text_access_token", return_value="test-token"),
            patch.object(web_search_tool.account_service, "mark_text_used"),
            patch.object(web_search_tool, "OpenAIBackendAPI", side_effect=build_backend),
        ):
            self.assertEqual(web_search_tool.run_web_search("test query"), {"answer": "ok"})

        self.assertIsNotNone(fake_backend)
        self.assertTrue(fake_backend.closed)


if __name__ == "__main__":
    unittest.main()
