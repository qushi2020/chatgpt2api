"""本地可靠背景消除（抠图）。

背景消除类请求（EDIT-05）不能交给 gpt-image 生成式编辑处理——那是整图重绘，
主体会挪位、尺寸变成模型固定档位、背景常常仍不透明。这里改用 rembg 判别式分割：
对**原图**产 alpha 蒙版，再贴回**原图像素**，保证输出与原图逐像素对齐、尺寸一致、
主体不挪位，透明或白底由 prompt 指定。

触发条件：图生图（edit）+ 有输入图 + prompt 命中抠图关键词。
"""

from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Any

from PIL import Image, ImageChops, ImageFilter

DEFAULT_REMBG_MODEL = "silueta"
_REMBG_SESSION: Any | None = None

# 命中任一关键词即视为抠图/背景消除请求；与 Pixora EDIT-05 的 prompt 用词对齐。
CUTOUT_KEYWORDS = (
    "背景分割",
    "抠图",
    "透明png",
    "透明底",
    "remove background",
    "background removal",
    "transparent background",
)

# 这些"动词+背景"短语只有在指整张背景时才算抠图请求；后面紧跟方位词
# （如"移除背景中的路人"）说明要移除的是背景里的东西、背景本身要保留，
# 这类 prompt（Pixora EDIT-07 消除路人等）必须走生成式编辑而不是抠图。
CONTEXTUAL_CUTOUT_KEYWORDS = (
    "去背景",
    "消除背景",
    "背景消除",
    "移除背景",
    "背景移除",
)

_BACKGROUND_LOCATIVE_SUFFIXES = ("中", "里", "上", "处", "内", "深")

# 用户在 Pixora 里选择的"输出背景"字段会渲染成 "输出背景：白底图 / 透明PNG"。
# 判定白底必须只看这个"选择行"，不能对整段 prompt 做子串搜索——EDIT-05 模板里有一句
# 同时解释透明和白底的静态说明（"…只把背景替换为纯白"、focus 里的"透明或白底效果"），
# 子串搜索会把它误判为白底，导致透明PNG（默认项）被错误输出成白底。
_OUTPUT_BACKGROUND_RE = re.compile(r"输出背景\s*[:：]\s*([^\n，,。;；]+)")
_WHITE_VALUE_TOKENS = ("白底", "纯白", "white")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def is_background_cutout_request(prompt: str, images: list[Any] | None) -> bool:
    """判断是否为需要本地抠图的背景消除请求。"""
    if not images:
        return False
    low = str(prompt or "").lower()
    if any(keyword in low for keyword in CUTOUT_KEYWORDS):
        return True
    for keyword in CONTEXTUAL_CUTOUT_KEYWORDS:
        start = 0
        while (idx := low.find(keyword, start)) != -1:
            tail = low[idx + len(keyword):]
            if not tail.startswith(_BACKGROUND_LOCATIVE_SUFFIXES):
                return True
            start = idx + len(keyword)
    return False


def wants_white_background_from_field(value: str) -> bool | None:
    """从显式 output_background 字段判断白底/透明；无法判断时返回 None。

    兼容 OpenAI 风格英文值（white/transparent 等）和 Pixora 中文选项（白底图/透明PNG）。
    """
    raw = str(value or "").strip()
    if not raw:
        return None
    low = raw.lower().replace("-", "_").replace(" ", "_")
    if low in {"white", "white_background", "solid_white", "opaque"} or "白底" in raw or "纯白" in raw:
        return True
    if low in {"transparent", "transparent_png"} or "透明" in raw:
        return False
    return None


def wants_white_background(prompt: str) -> bool:
    """从 prompt 判断输出白底图还是透明 PNG（默认透明）。

    优先读取 "输出背景：<选择>" 这一行的用户实际选择；只有在没有该选择行的
    非 Pixora 调用场景，才回退到宽松判定（且要求提到白底而未提透明，避免误判）。
    """
    text = str(prompt or "")
    match = _OUTPUT_BACKGROUND_RE.search(text)
    if match:
        value = match.group(1).lower()
        return any(token in value for token in _WHITE_VALUE_TOKENS)
    low = text.lower()
    if "white background" in low or "solid white" in low:
        return True
    has_white = ("白底" in text) or ("纯白" in text)
    has_transparent = ("透明" in text)
    return has_white and not has_transparent


def _rembg_session() -> Any:
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    try:
        from rembg import new_session
    except Exception as exc:  # pragma: no cover - 依赖缺失时的清晰报错
        raise RuntimeError("本地扣图组件 rembg 未安装，无法执行可靠背景消除") from exc

    model_name = os.environ.get("CHATGPT2API_REMBG_MODEL", DEFAULT_REMBG_MODEL).strip()
    try:
        _REMBG_SESSION = new_session(model_name) if model_name else new_session()
    except Exception as exc:
        raise RuntimeError(f"本地扣图模型初始化失败: {exc}") from exc
    return _REMBG_SESSION


def _rembg_remove(original_data: bytes) -> Image.Image:
    try:
        from rembg import remove
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("本地扣图组件 rembg 未安装，无法执行可靠背景消除") from exc

    try:
        output = remove(
            original_data,
            session=_rembg_session(),
            alpha_matting=_env_bool("CHATGPT2API_REMBG_ALPHA_MATTING", False),
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            # post_process_mask 会把蒙版硬二值化（<127→0，≥127→255）。silueta 在
            # 320×320 推理，蒙版放大回原尺寸后再二值化就是边缘锯齿的来源，默认关闭，
            # 交给 _refine_alpha 做平滑边缘精修。
            post_process_mask=_env_bool("CHATGPT2API_REMBG_POST_PROCESS", False),
        )
    except Exception as exc:
        raise RuntimeError(f"本地扣图失败: {exc}") from exc

    try:
        if isinstance(output, Image.Image):
            return output.convert("RGBA")
        if isinstance(output, (bytes, bytearray)):
            return Image.open(BytesIO(output)).convert("RGBA")
    except Exception as exc:
        raise RuntimeError(f"本地扣图结果无法读取: {exc}") from exc
    raise RuntimeError("本地扣图结果格式不受支持")


def _refine_alpha(alpha: Image.Image) -> Image.Image:
    """边缘精修：收窄过渡带 + 收边去彩边 + 轻羽化。

    低分辨率蒙版 LANCZOS 放大后边缘有一条较宽的软过渡带（直接用会显糊），
    但硬二值化又会产生锯齿。这里取中间路线：
    1. 线性收窄过渡带（低于 low 视为背景、高于 high 视为主体，中间线性拉伸），
       同时清掉背景上的弱噪声；
    2. MinFilter 收边 1px，切掉边缘沾染背景色的彩边（fringe）；
    3. 小半径高斯羽化，得到 1-2px 的平滑抗锯齿过渡。
    """
    low = max(0.0, min(254.0, _env_float("CHATGPT2API_CUTOUT_BAND_LOW", 64.0)))
    high = max(low + 1.0, min(255.0, _env_float("CHATGPT2API_CUTOUT_BAND_HIGH", 192.0)))
    scale = 255.0 / (high - low)
    alpha = alpha.point([max(0, min(255, round((v - low) * scale))) for v in range(256)])
    erode_px = int(_env_float("CHATGPT2API_CUTOUT_ERODE_PX", 1.0))
    for _ in range(max(0, erode_px)):
        alpha = alpha.filter(ImageFilter.MinFilter(3))
    feather = _env_float("CHATGPT2API_CUTOUT_FEATHER_PX", 1.2)
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    return alpha


def _alpha_from_original(original_data: bytes, original_size: tuple[int, int]) -> Image.Image:
    cutout = _rembg_remove(original_data)
    alpha = cutout.getchannel("A")
    if alpha.size != original_size:
        alpha = alpha.resize(original_size, Image.LANCZOS)
    alpha = _refine_alpha(alpha)
    min_alpha, max_alpha = alpha.getextrema()
    if min_alpha >= 250:
        raise RuntimeError("本地扣图未产生透明背景，请换一张主体更清晰的图片再试")
    if max_alpha <= 5:
        raise RuntimeError("本地扣图未识别到可保留主体，请换一张主体更清晰的图片再试")
    return alpha


def _compose_original_cutout(original_data: bytes, alpha: Image.Image, white_background: bool) -> bytes:
    original = Image.open(BytesIO(original_data)).convert("RGBA")
    if alpha.size != original.size:
        alpha = alpha.resize(original.size, Image.LANCZOS)
    source_alpha = original.getchannel("A")
    if source_alpha.getextrema()[0] < 255:
        alpha = ImageChops.multiply(alpha, source_alpha)
    original.putalpha(alpha)
    if white_background:
        white = Image.new("RGBA", original.size, (255, 255, 255, 255))
        white.alpha_composite(original)
        original = white
    buf = BytesIO()
    original.save(buf, format="PNG")
    return buf.getvalue()


def cutout_image_bytes(original_data: bytes, white_background: bool) -> bytes:
    """对单张图执行背景消除，返回原尺寸、主体对齐的 PNG 字节。"""
    original = Image.open(BytesIO(original_data)).convert("RGBA")
    alpha = _alpha_from_original(original_data, original.size)
    return _compose_original_cutout(original_data, alpha, white_background)
