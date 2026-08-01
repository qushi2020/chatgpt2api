import unittest

from services.protocol.conversation import final_image_prompt


class ImagePromptTests(unittest.TestCase):
    def test_verbatim_image_prompt_does_not_append_size_or_quality(self):
        prompt = "修复老照片，渲染为彩色。"

        self.assertEqual(final_image_prompt(prompt, "1536x1205", "high", True), prompt)

    def test_default_image_prompt_keeps_legacy_size_and_quality_hints(self):
        prompt = final_image_prompt("draw a cat", "1024x1024", "high", False)

        self.assertIn("draw a cat", prompt)
        self.assertIn("输出图片尺寸为 1024x1024。", prompt)
        self.assertIn("输出图片质量为 high。", prompt)


if __name__ == "__main__":
    unittest.main()
