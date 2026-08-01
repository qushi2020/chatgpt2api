from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any, Iterator

from PIL import Image

from services.protocol.background_cutout import (
    cutout_image_bytes,
    is_background_cutout_request,
    wants_white_background,
    wants_white_background_from_field,
)
from services.protocol.conversation import (
    ConversationRequest,
    ImageGenerationError,
    collect_image_outputs,
    count_text_tokens,
    encode_images,
    format_image_result,
    stream_image_chunks,
    stream_image_outputs_with_pool,
)
from utils.image_tokens import count_image_inputs_tokens, count_image_output_items_tokens, image_usage


def _composite_mask(
    images: list[tuple[bytes, str, str]],
    masks: list[tuple[bytes, str, str]],
) -> list[tuple[bytes, str, str]]:
    """将 mask 的 alpha 通道合成到图片中，标识需要编辑的区域。

    mask 的透明区域（低 alpha）= 需要编辑的区域，
    mask 的不透明区域（高 alpha）= 保留的区域。
    如果无 mask 则返回原图。
    """
    if not masks:
        return images
    result: list[tuple[bytes, str, str]] = []
    for i, (data, filename, mime_type) in enumerate(images):
        mask_data = masks[i][0] if i < len(masks) else masks[-1][0]
        img = Image.open(BytesIO(data)).convert("RGBA")
        mask_img = Image.open(BytesIO(mask_data))
        if mask_img.mode == "RGBA":
            alpha = mask_img.split()[3]
        elif mask_img.mode == "L":
            alpha = mask_img
        else:
            alpha = mask_img.convert("L")
        alpha = alpha.resize(img.size, Image.LANCZOS)
        img.putalpha(alpha)
        buf = BytesIO()
        img.save(buf, format="PNG")
        result.append((buf.getvalue(), filename, "image/png"))
    return result


def _local_background_cutout(
    images: list[tuple[bytes, str, str]],
    *,
    prompt: str,
    model: str,
    size: Any,
    quality: str,
    response_format: str,
    base_url: str | None,
    progress_callback: Any = None,
    output_background: str = "",
) -> dict[str, Any]:
    """本地 rembg 抠图并组装成与正常路径一致的结果。"""
    if callable(progress_callback):
        try:
            progress_callback("local_background_cutout")
        except Exception:
            pass
    # 显式 output_background 字段优先；无字段时回退 prompt 文案判定。
    white_from_field = wants_white_background_from_field(output_background)
    white_background = white_from_field if white_from_field is not None else wants_white_background(prompt)
    items: list[dict[str, Any]] = []
    for data, _filename, _mime in images:
        png = cutout_image_bytes(data, white_background)
        items.append({"b64_json": base64.b64encode(png).decode("ascii"), "revised_prompt": prompt})
    result = format_image_result(items, prompt, response_format, base_url, int(time.time()))
    if not result.get("data"):
        raise ImageGenerationError("本地扣图未生成有效结果")
    result["usage"] = image_usage(
        input_text_tokens=count_text_tokens(prompt, model),
        input_image_tokens=count_image_inputs_tokens(images, model),
        output_tokens=count_image_output_items_tokens(result.get("data"), size, quality),
    )
    return result


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    prompt = str(body.get("prompt") or "")
    images = body.get("images") or []
    masks = body.get("mask") or []
    model = str(body.get("model") or "gpt-image-2")
    n = int(body.get("n") or 1)
    size = body.get("size")
    quality = str(body.get("quality") or "auto")
    verbatim_prompt = bool(body.get("verbatim_prompt"))
    response_format = str(body.get("response_format") or "b64_json")
    base_url = str(body.get("base_url") or "") or None
    progress_callback = body.get("progress_callback")
    task_type = str(body.get("task_type") or "").strip().lower().replace("-", "_")
    output_background = str(body.get("output_background") or "")
    # 背景消除（EDIT-05 等）走本地 rembg 抠图：对原图产 alpha 贴回原图，
    # 原尺寸、主体像素级对齐、透明/白底可控，绕开 gpt-image 生成式重绘。
    # 判定优先级：显式 task_type 字段 > prompt 关键词兜底。
    # 调用方显式传了非抠图 task_type（如 edit）时，禁用关键词兜底，
    # 避免 prompt 里偶然出现"抠图/透明PNG"等字样把生成式编辑请求整体劫持成抠图。
    explicit_cutout = task_type in {"background_removal", "cutout", "matting"}
    keyword_cutout = not task_type and is_background_cutout_request(prompt, images)
    if (explicit_cutout and images) or keyword_cutout:
        return _local_background_cutout(
            images,
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            response_format=response_format,
            base_url=base_url,
            progress_callback=progress_callback,
            output_background=output_background,
        )
    images = _composite_mask(images, masks)
    encoded_images = encode_images(images)
    if not encoded_images:
        raise ImageGenerationError("image is required")
    outputs = stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        n=n,
        size=size,
        quality=quality,
        verbatim_prompt=verbatim_prompt,
        response_format=response_format,
        base_url=base_url,
        images=encoded_images,
        message_as_error=True,
        progress_callback=progress_callback,
    ))
    if body.get("stream"):
        return stream_image_chunks(outputs)
    result = collect_image_outputs(outputs)
    result["usage"] = image_usage(
        input_text_tokens=count_text_tokens(prompt, model),
        input_image_tokens=count_image_inputs_tokens(images, model),
        output_tokens=count_image_output_items_tokens(result.get("data"), size, quality),
    )
    return result
