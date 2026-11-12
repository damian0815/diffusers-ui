"""Unit tests for _save_image output-format handling (jpg vs png)."""
import pytest
from PIL import Image as PILImage

from backend.config import settings
from backend.pipeline.worker import _save_image


@pytest.fixture
def image_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "images_dir", tmp_path / "images")
    monkeypatch.setattr(settings, "thumbnails_dir", tmp_path / "images" / "thumbs")
    return tmp_path / "images", tmp_path / "images" / "thumbs"


def test_jpg_is_default(image_dirs):
    images_dir, thumbs_dir = image_dirs
    name = _save_image(PILImage.new("RGB", (100, 80), "red"))

    assert name.endswith(".jpg")
    full = images_dir / name
    thumb = thumbs_dir / name
    assert full.exists() and thumb.exists()
    with PILImage.open(full) as img:
        assert img.format == "JPEG"
    with PILImage.open(thumb) as img:
        assert max(img.size) <= max(settings.thumbnail_size)


def test_jpg_converts_mode_with_alpha(image_dirs):
    """JPEG cannot store alpha; _save_image must convert instead of raising."""
    images_dir, _ = image_dirs
    name = _save_image(PILImage.new("RGBA", (64, 64), (0, 0, 255, 128)))

    with PILImage.open(images_dir / name) as img:
        assert img.mode == "RGB"


def test_png_format_still_supported(image_dirs, monkeypatch):
    monkeypatch.setattr(settings, "image_format", "png")
    images_dir, thumbs_dir = image_dirs
    name = _save_image(PILImage.new("RGB", (64, 64), "blue"))

    assert name.endswith(".png")
    with PILImage.open(images_dir / name) as img:
        assert img.format == "PNG"
