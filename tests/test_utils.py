import pytest
from PIL import Image

from llmsuite.utils import (
    encode_image,
    format_anthropic_image_content,
    format_openai_image_content,
)


@pytest.fixture
def jpeg_image_path(tmp_path):
    # Create a simple JPEG image in memory
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    return img_path


@pytest.fixture
def png_image_path(tmp_path):
    # Create a simple PNG image in memory
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    img_path = tmp_path / "test.png"
    img.save(img_path)
    return img_path


def test_encode_image_jpeg(jpeg_image_path):
    media_type, base64_img = encode_image(jpeg_image_path)

    assert media_type == "image/jpeg"
    assert base64_img.startswith("iVBOR") or base64_img.startswith(
        "/9j/"
    )  # Common JPEG/PNG starts
    assert isinstance(base64_img, str)


def test_encode_image_png(png_image_path):
    media_type, base64_img = encode_image(png_image_path)

    assert media_type == "image/png"
    assert base64_img.startswith("iVBOR") or base64_img.startswith(
        "/9j/"
    )  # Common JPEG/PNG starts
    assert isinstance(base64_img, str)


def test_encode_image_unsupported_format(tmp_path):
    # Create a text file with .gif extension
    gif_path = tmp_path / "test.gif"
    gif_path.write_text("Not a real GIF")

    with pytest.raises(ValueError, match="Unsupported image format"):
        encode_image(gif_path)


def test_format_openai_image_content(jpeg_image_path):
    result = format_openai_image_content("Test message", jpeg_image_path)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "Test message"}
    assert result[1]["type"] == "image_url"
    assert result[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_format_anthropic_image_content(png_image_path):
    result = format_anthropic_image_content("Test message", png_image_path)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "Test message"}
    assert result[1]["type"] == "image"
    assert result[1]["source"]["type"] == "base64"
    assert result[1]["source"]["media_type"] == "image/png"
    assert isinstance(result[1]["source"]["data"], str)
