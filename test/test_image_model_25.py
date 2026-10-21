from __future__ import annotations

import json
import unittest
from unittest import mock

from services.account_service import account_service
from services.config import config
from services.openai_backend_api import OpenAIBackendAPI
from utils.helper import (
    CODEX_IMAGE_25_MODEL,
    CODEX_IMAGE_25_SUNBURST_MODEL,
    is_codex_image_model,
    is_supported_image_model,
    split_image_model,
)


class FakeCodexRawResponse:
    def __init__(self, text: str = "", content_type: str = "text/event-stream") -> None:
        self.headers = {"content-type": content_type}
        self.status = 200
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")

    def __enter__(self) -> "FakeCodexRawResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class SplitImageModel25Tests(unittest.TestCase):
    def test_web_image_25_alias(self) -> None:
        self.assertEqual(split_image_model("gpt-image-2.5"), (None, "gpt-image-2.5"))
        self.assertTrue(is_supported_image_model("gpt-image-2.5"))
        self.assertFalse(is_codex_image_model("gpt-image-2.5"))
        for prefix in ("plus", "team", "pro"):
            self.assertEqual(split_image_model(f"{prefix}-gpt-image-2.5"), (None, None))

    def test_codex_image_25_aliases(self) -> None:
        self.assertEqual(split_image_model("codex-gpt-image-2.5"), (None, "codex-gpt-image-2.5"))
        self.assertEqual(
            split_image_model("codex-gpt-image-2.5-sunburst"),
            (None, "codex-gpt-image-2.5-sunburst"),
        )
        for prefix in ("plus", "team", "pro"):
            self.assertEqual(
                split_image_model(f"{prefix}-codex-gpt-image-2.5"),
                (prefix, "codex-gpt-image-2.5"),
            )
            self.assertEqual(
                split_image_model(f"{prefix}-codex-gpt-image-2.5-sunburst"),
                (prefix, "codex-gpt-image-2.5-sunburst"),
            )
        self.assertTrue(is_codex_image_model(CODEX_IMAGE_25_MODEL))
        self.assertTrue(is_codex_image_model(CODEX_IMAGE_25_SUNBURST_MODEL))


class ImageModelSettings25Tests(unittest.TestCase):
    def test_web_image_25_uses_configured_slug(self) -> None:
        original_value = config.data.get("web_image_model_slug")
        backend = OpenAIBackendAPI()
        try:
            config.data["web_image_model_slug"] = "gpt-5-6-free"
            self.assertEqual(backend._image_model_settings("gpt-image-2.5")[0], "gpt-5-6-free")
            self.assertEqual(backend._image_model_settings("gpt-image-2")[0], "gpt-5-6-free")
        finally:
            if original_value is None:
                config.data.pop("web_image_model_slug", None)
            else:
                config.data["web_image_model_slug"] = original_value
            backend.close()

    def test_codex_image_25_passes_alias_through(self) -> None:
        backend = OpenAIBackendAPI()
        try:
            self.assertEqual(
                backend._image_model_settings("codex-gpt-image-2.5")[0],
                "codex-gpt-image-2.5",
            )
            self.assertEqual(
                backend._image_model_settings("codex-gpt-image-2.5-sunburst")[0],
                "codex-gpt-image-2.5-sunburst",
            )
        finally:
            backend.close()


class CodexImageToolModelTests(unittest.TestCase):
    def _tool_model_for(self, model: str) -> str:
        backend = OpenAIBackendAPI(access_token="token")
        try:
            with (
                mock.patch.object(
                    account_service,
                    "get_account",
                    return_value={"source_type": "codex", "type": "Plus"},
                ),
                mock.patch(
                    "urllib.request.urlopen",
                    return_value=FakeCodexRawResponse("data: [DONE]\n\n"),
                ) as urlopen_mock,
            ):
                list(backend.iter_codex_image_response_events(
                    prompt="test", model=model,
                ))
            request = urlopen_mock.call_args[0][0]
            body = json.loads(request.data.decode("utf-8"))
            return body["tools"][0]["model"]
        finally:
            backend.close()

    def test_codex_25_maps_to_flare(self) -> None:
        self.assertEqual(self._tool_model_for("codex-gpt-image-2.5"), "gpt-image-2.5-flare")
        self.assertEqual(
            self._tool_model_for("plus-codex-gpt-image-2.5"),
            "gpt-image-2.5-flare",
        )

    def test_codex_25_sunburst_maps_to_sunburst(self) -> None:
        self.assertEqual(
            self._tool_model_for("codex-gpt-image-2.5-sunburst"),
            "gpt-image-2.5-sunburst",
        )

    def test_codex_image_2_keeps_legacy_model(self) -> None:
        self.assertEqual(self._tool_model_for("codex-gpt-image-2"), "gpt-image-2")

    def test_default_model_keeps_legacy_model(self) -> None:
        self.assertEqual(self._tool_model_for(""), "gpt-image-2")


if __name__ == "__main__":
    unittest.main()
