import base64
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from services.config import config
from services.openai_backend_api import ChatRequirements, OpenAIBackendAPI
from utils.helper import UpstreamHTTPError


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload or {}
        self.status_code = status_code
        self.headers = {}
        self.text = ""

    def json(self):
        return self._payload


class CaptureSession:
    def __init__(self):
        self.posts = []
        self.puts = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if url.endswith("/backend-api/f/conversation/prepare"):
            return FakeResponse({"conduit_token": "conduit"})
        if url.endswith("/backend-api/files"):
            return FakeResponse({
                "file_id": "file_123",
                "library_file_id": "library_123",
                "upload_url": "https://upload.invalid/blob",
            })
        return FakeResponse()

    def put(self, url, **kwargs):
        self.puts.append((url, kwargs))
        return FakeResponse()


class BootstrapSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.headers = {
            "Sec-Ch-Ua": '"Chromium";v="143"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
        }

    def get(self, url, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


def backend_with_capture():
    backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
    backend.base_url = "https://chatgpt.com"
    backend.account = {"default_model_slug": "gpt-5-6-pro"}
    backend.session = CaptureSession()
    backend.user_agent = "test-agent"
    backend._image_headers = lambda *args, **kwargs: {}
    backend._headers = lambda *args, **kwargs: {}
    return backend


class ImageWebTurnTests(unittest.TestCase):
    def test_prepare_and_final_share_the_same_multimodal_turn(self):
        original_model_slug = config.data.get("web_image_model_slug")
        original_thinking_effort = config.data.get("image_thinking_effort")
        backend = backend_with_capture()
        try:
            config.data["web_image_model_slug"] = "gpt-5-6-free"
            config.data["image_thinking_effort"] = "high"
            reference = {
                "file_id": "file_123",
                "library_file_id": "library_123",
                "file_name": "source.jpg",
                "file_size": 12345,
                "mime_type": "image/jpeg",
                "width": 700,
                "height": 549,
            }
            requirements = ChatRequirements(token="requirements")

            conduit = backend._prepare_image_conversation(
                "修复老照片，渲染为彩色。",
                requirements,
                "gpt-image-2",
                [reference],
                "message-1",
                "client-created-root",
                123.5,
            )
            backend._start_image_generation(
                "修复老照片，渲染为彩色。",
                requirements,
                conduit,
                "gpt-image-2",
                [reference],
                "message-1",
                "client-created-root",
                123.5,
            )
        finally:
            if original_model_slug is None:
                config.data.pop("web_image_model_slug", None)
            else:
                config.data["web_image_model_slug"] = original_model_slug
            if original_thinking_effort is None:
                config.data.pop("image_thinking_effort", None)
            else:
                config.data["image_thinking_effort"] = original_thinking_effort

        prepare_payload = backend.session.posts[0][1]["json"]
        final_payload = backend.session.posts[1][1]["json"]
        prepared_message = prepare_payload["partial_query"]
        final_message = final_payload["messages"][0]

        self.assertEqual(conduit, "conduit")
        self.assertEqual(prepare_payload["parent_message_id"], "client-created-root")
        self.assertEqual(final_payload["parent_message_id"], "client-created-root")
        self.assertEqual(prepared_message, final_message)
        self.assertEqual(prepared_message["id"], "message-1")
        self.assertEqual(prepared_message["content"]["content_type"], "multimodal_text")
        self.assertEqual(
            prepared_message["content"]["parts"][0]["asset_pointer"],
            "sediment://file_123",
        )
        self.assertEqual(prepare_payload["attachment_mime_types"], ["image/jpeg"])
        attachment = prepared_message["metadata"]["attachments"][0]
        self.assertEqual(attachment["mime_type"], "image/jpeg")
        self.assertEqual(attachment["source"], "library")
        self.assertEqual(attachment["library_file_id"], "library_123")
        self.assertEqual(prepare_payload["model"], "gpt-5-6-free")
        self.assertEqual(final_payload["model"], "gpt-5-6-free")
        self.assertEqual(prepare_payload["client_prepare_state"], "none")
        self.assertEqual(final_payload["client_prepare_state"], "success")
        self.assertEqual(prepare_payload["thinking_effort"], "high")
        self.assertEqual(final_payload["thinking_effort"], "high")
        self.assertNotIn("force_parallel_switch", final_payload)
        self.assertNotIn("paragen_cot_summary_display_override", final_payload)

    def test_asset_pointer_matches_web_file_id_scheme(self):
        self.assertEqual(
            OpenAIBackendAPI._image_asset_pointer({"file_id": "file_123"}),
            "sediment://file_123",
        )
        self.assertEqual(
            OpenAIBackendAPI._image_asset_pointer({"file_id": "legacy-file-123"}),
            "file-service://legacy-file-123",
        )

    def test_upload_preserves_real_jpeg_type_and_uses_library_storage(self):
        backend = backend_with_capture()
        image = Image.new("RGB", (7, 5), color=(120, 120, 120))
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        reference = backend._upload_image(
            f"data:image/jpeg;base64,{encoded}",
            "image_1",
        )

        upload_payload = backend.session.posts[0][1]["json"]
        self.assertEqual(upload_payload["file_name"], "image_1.jpg")
        self.assertEqual(upload_payload["width"], 7)
        self.assertEqual(upload_payload["height"], 5)
        self.assertTrue(upload_payload["store_in_library"])
        self.assertEqual(upload_payload["library_persistence_mode"], "opportunistic")
        self.assertEqual(reference["file_name"], "image_1.jpg")
        self.assertEqual(reference["mime_type"], "image/jpeg")
        self.assertEqual(reference["library_file_id"], "library_123")

    def test_bootstrap_retries_transient_503_before_succeeding(self):
        first = FakeResponse(status_code=503)
        first.text = "upstream connect error or disconnect/reset before headers"
        second = FakeResponse(status_code=200)
        second.text = '<script src="/_next/static/chunks/main.js"></script>'
        backend = backend_with_capture()
        backend.session = BootstrapSession([first, second])

        with patch("services.openai_backend_api.time.sleep") as sleep:
            backend._bootstrap()

        self.assertEqual(backend.session.calls, 2)
        sleep.assert_called_once_with(1.0)

    def test_bootstrap_does_not_retry_non_transient_failure(self):
        response = FakeResponse(status_code=401)
        response.text = "unauthorized"
        backend = backend_with_capture()
        backend.session = BootstrapSession([response])

        with patch("services.openai_backend_api.time.sleep") as sleep:
            with self.assertRaises(UpstreamHTTPError):
                backend._bootstrap()

        self.assertEqual(backend.session.calls, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
