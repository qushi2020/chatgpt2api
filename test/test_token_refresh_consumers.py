from __future__ import annotations

import unittest
from unittest.mock import patch

from services.account_service import TokenRefreshResult
from services.protocol import conversation


class FakeBackend:
    def __init__(self, access_token: str = "token") -> None:
        self.access_token = access_token
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TokenRefreshConsumerTests(unittest.TestCase):
    def test_text_stream_preserves_account_on_transient_refresh_failure(self) -> None:
        outer_backend = FakeBackend()
        request = conversation.ConversationRequest(prompt="hello")

        with (
            patch.object(conversation, "OpenAIBackendAPI", side_effect=lambda access_token="": FakeBackend(access_token)),
            patch.object(
                conversation,
                "conversation_events",
                side_effect=RuntimeError("token_invalidated"),
            ),
            patch.object(
                conversation.account_service,
                "refresh_access_token_result",
                return_value=TokenRefreshResult(
                    "token",
                    attempted=True,
                    error="OAuth request timed out",
                    error_kind="transient",
                ),
            ),
            patch.object(conversation.account_service, "get_text_access_token", return_value=""),
            patch.object(
                conversation.account_service,
                "confirm_invalid_access_token",
                return_value=False,
            ) as confirm_invalid,
            patch.object(conversation.account_service, "remove_invalid_token") as remove_invalid,
        ):
            with self.assertRaisesRegex(RuntimeError, "token_invalidated"):
                list(conversation.stream_text_deltas(outer_backend, request))

        remove_invalid.assert_not_called()
        confirm_invalid.assert_not_called()
        self.assertTrue(outer_backend.closed)

    def test_text_stream_preserves_account_without_refresh_token(self) -> None:
        backend = FakeBackend("token")
        request = conversation.ConversationRequest(prompt="hello")

        with (
            patch.object(
                conversation,
                "conversation_events",
                side_effect=RuntimeError("token_invalidated"),
            ),
            patch.object(
                conversation.account_service,
                "refresh_access_token_result",
                return_value=TokenRefreshResult("token"),
            ),
            patch.object(conversation.account_service, "get_text_access_token", return_value=""),
            patch.object(
                conversation.account_service,
                "confirm_invalid_access_token",
                return_value=False,
            ) as confirm_invalid,
            patch.object(conversation.account_service, "remove_invalid_token") as remove_invalid,
        ):
            with self.assertRaisesRegex(RuntimeError, "token_invalidated"):
                list(conversation.stream_text_deltas(backend, request))

        remove_invalid.assert_not_called()
        confirm_invalid.assert_called_once_with("token", "text_stream")
        self.assertTrue(backend.closed)

    def test_image_stream_preserves_account_on_transient_refresh_failure(self) -> None:
        request = conversation.ConversationRequest(model="codex-gpt-image-2", prompt="draw")

        with (
            patch.object(conversation.account_service, "get_available_access_token", return_value="token"),
            patch.object(
                conversation.account_service,
                "get_account",
                return_value={"access_token": "token", "source_type": "codex"},
            ),
            patch.object(conversation, "OpenAIBackendAPI", side_effect=lambda access_token="": FakeBackend(access_token)),
            patch.object(
                conversation,
                "stream_codex_image_outputs",
                side_effect=RuntimeError("token_invalidated"),
            ),
            patch.object(conversation.account_service, "mark_image_result"),
            patch.object(
                conversation.account_service,
                "refresh_access_token_result",
                return_value=TokenRefreshResult(
                    "token",
                    attempted=True,
                    error="OAuth request timed out",
                    error_kind="transient",
                ),
            ),
            patch.object(conversation.account_service, "remove_invalid_token") as remove_invalid,
        ):
            with self.assertRaisesRegex(RuntimeError, "token_invalidated"):
                conversation._generate_single_image(request, 1, 1)

        remove_invalid.assert_not_called()

    def test_image_stream_preserves_account_without_refresh_token(self) -> None:
        backend = FakeBackend("token")
        request = conversation.ConversationRequest(
            model="gpt-image-2",
            prompt="test",
            size="1024x1024",
            quality="auto",
        )

        with (
            patch.object(conversation.account_service, "get_available_access_token", return_value="token"),
            patch.object(
                conversation.account_service,
                "get_account",
                return_value={
                    "access_token": "token",
                    "source_type": "web",
                    "status": "正常",
                    "quota": 1,
                },
            ),
            patch.object(
                conversation.account_service,
                "refresh_access_token_result",
                return_value=TokenRefreshResult("token"),
            ),
            patch.object(conversation.account_service, "mark_image_result"),
            patch.object(
                conversation.account_service,
                "confirm_invalid_access_token",
                return_value=False,
            ) as confirm_invalid,
            patch.object(conversation.account_service, "remove_invalid_token") as remove_invalid,
            patch.object(conversation, "OpenAIBackendAPI", return_value=backend),
            patch.object(
                conversation,
                "stream_image_outputs",
                side_effect=RuntimeError("token_invalidated"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "token_invalidated"):
                conversation._generate_single_image(request, 1, 1)

        remove_invalid.assert_not_called()
        confirm_invalid.assert_called_once_with("token", "image_stream")


if __name__ == "__main__":
    unittest.main()
