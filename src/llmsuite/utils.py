import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal

from PIL import Image

type MessageRole = Literal["system", "user", "assistant"]


def get_base64_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def encode_image(
    image_path: Path, max_size: tuple[int, int] = (1024, 1024), quality: int = 85
) -> tuple[str, str]:
    if image_path.suffix.lower() in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif image_path.suffix.lower() == ".png":
        media_type = "image/png"
    else:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    img = Image.open(image_path)

    if media_type == "image/jpeg" and img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail(max_size)
    buffer = BytesIO()
    fmt = "JPEG" if media_type == "image/jpeg" else "PNG"
    img.save(buffer, format=fmt, quality=quality, optimize=True)
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return media_type, base64_image


def format_openai_image_content(text: str, image_path: Path) -> List[Dict[str, Any]]:
    media_type, base64_image = encode_image(image_path)
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
        },
    ]


def format_anthropic_image_content(text: str, image_path: Path) -> List[Dict[str, Any]]:
    media_type, base64_image = encode_image(image_path)
    return [
        {"type": "text", "text": text},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_image,
            },
        },
    ]
