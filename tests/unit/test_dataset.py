"""Unit tests for backend/training/dataset.py"""
import pytest
from PIL import Image

from backend.training.aspects import compute_resolution_options, get_aspect_buckets
from backend.training.dataset import TextualInversionDataset


# ---------------------------------------------------------------------------
# compute_resolution_options integration tests
# ---------------------------------------------------------------------------

def test_square_image_gets_square_bucket():
    buckets = get_aspect_buckets(512)
    result = compute_resolution_options(512, 512, {512: buckets})
    assert result[512].target_wh == [512, 512]


def test_landscape_4_3_gets_wide_bucket():
    buckets = get_aspect_buckets(512)
    result = compute_resolution_options(800, 600, {512: buckets})
    tw, th = result[512].target_wh
    assert tw > th


def test_portrait_3_4_gets_tall_bucket():
    buckets = get_aspect_buckets(512)
    result = compute_resolution_options(600, 800, {512: buckets})
    tw, th = result[512].target_wh
    assert th > tw


def test_ultrawide_gets_extreme_bucket():
    buckets = get_aspect_buckets(512)
    result = compute_resolution_options(2000, 500, {512: buckets})
    tw, th = result[512].target_wh
    assert tw / th > 2.0


# ---------------------------------------------------------------------------
# Dataset tests (require image files on disk)
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal tokenizer stand-in that returns fixed-length input_ids."""
    model_max_length = 77

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        ids = list(range(len(text.split())))
        ids = ids[:max_length]
        ids += [0] * (max_length - len(ids))
        attention_mask = [1 if i < len(text.split()) else 0 for i in range(max_length)]

        import torch
        class _Out:
            def __init__(self, ids, attention_mask):
                self.input_ids = torch.tensor([ids])
                self.attention_mask = torch.tensor([attention_mask])
        return _Out(ids, attention_mask)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def _make_image(path, w, h, color=(128, 64, 32)):
    Image.new("RGB", (w, h), color).save(path)


@pytest.fixture
def sample_images(tmp_path):
    """Create a set of synthetic images with different aspect ratios."""
    images = {
        "square.png": (512, 512),
        "landscape.png": (800, 600),
        "portrait.png": (600, 800),
        "ultrawide.png": (1920, 540),
    }
    for name, (w, h) in images.items():
        _make_image(tmp_path / name, w, h)
    return list(images.keys())


@pytest.fixture
def mock_images_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.training.dataset.settings",
        type("S", (), {"images_dir": tmp_path})(),
    )


def test_dataset_output_shape(sample_images, mock_images_dir):
    ds = TextualInversionDataset(
        sample_images, _FakeTokenizer(), "<sks>", size=512, repeats=1,
    )
    item = ds[0]
    assert item["pixel_values"].shape == (3, 512, 512)
    assert item["input_ids"].shape == (77,)


def test_dataset_repeats(sample_images, mock_images_dir):
    ds = TextualInversionDataset(
        sample_images, _FakeTokenizer(), "<sks>", size=512, repeats=5,
    )
    assert len(ds) == len(sample_images) * 5


def test_class_prompt_in_text(sample_images, mock_images_dir):
    ds = TextualInversionDataset(
        sample_images, _FakeTokenizer(), "<sks>", size=512, repeats=1,
        class_prompt="dog",
    )
    for i in range(len(sample_images)):
        path, tw, th = ds._entries[i]
        assert ds.class_prompt == "dog"


def test_no_class_prompt(sample_images, mock_images_dir):
    ds = TextualInversionDataset(
        sample_images, _FakeTokenizer(), "<sks>", size=512, repeats=1,
        class_prompt="",
    )
    assert ds.class_prompt == ""


def test_dataset_all_ratios_produce_square_crop(sample_images, mock_images_dir):
    ds = TextualInversionDataset(
        sample_images, _FakeTokenizer(), "<sks>", size=512, repeats=1,
    )
    for i in range(len(sample_images)):
        item = ds[i]
        assert item["pixel_values"].shape == (3, 512, 512), (
            f"Image {i} produced shape {item['pixel_values'].shape}"
        )


def test_entries_use_compute_resolution_options(sample_images, mock_images_dir):
    ds = TextualInversionDataset(
        sample_images, _FakeTokenizer(), "<sks>", size=512, repeats=1,
    )
    # Verify entries were computed via compute_resolution_options:
    # resize dims must have short side == 512
    for path, tw, th in ds._entries:
        assert min(tw, th) == 512, (
            f"Short side of ({tw}, {th}) should be 512"
        )


# ---------------------------------------------------------------------------
# Token expansion tests
# ---------------------------------------------------------------------------

class _RecordingTokenizer(_FakeTokenizer):
    """Tokenizer that records the text it receives for later inspection."""

    def __init__(self):
        self.last_text = None

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        self.last_text = text
        return super().__call__(text, padding, truncation, max_length, return_tensors)


def test_single_vector_expansion_in_dataset(sample_images, mock_images_dir):
    tok = _RecordingTokenizer()
    ds = TextualInversionDataset(
        sample_images, tok, "<sks>", size=512, repeats=1, num_vectors=1,
    )
    ds[0]
    assert "sks.0" in tok.last_text
    assert "<sks>" not in tok.last_text


def test_multi_vector_expansion_in_dataset(sample_images, mock_images_dir):
    tok = _RecordingTokenizer()
    ds = TextualInversionDataset(
        sample_images, tok, "<sks>", size=512, repeats=1, num_vectors=3,
    )
    ds[0]
    assert "sks.0 sks.1 sks.2" in tok.last_text


def test_expansion_with_class_prompt(sample_images, mock_images_dir):
    tok = _RecordingTokenizer()
    ds = TextualInversionDataset(
        sample_images, tok, "<sks>", size=512, repeats=1,
        num_vectors=2, class_prompt="dog",
    )
    ds[0]
    assert "sks.0 sks.1 dog" in tok.last_text


def test_no_expansion_when_num_vectors_one(sample_images, mock_images_dir):
    tok = _RecordingTokenizer()
    ds = TextualInversionDataset(
        sample_images, tok, "<sks>", size=512, repeats=1, num_vectors=1,
    )
    ds[0]
    assert "sks.0" in tok.last_text
    assert "sks.1" not in tok.last_text


# ---------------------------------------------------------------------------
# Rotation augmentation tests
# ---------------------------------------------------------------------------

def test_compute_max_rotation_angle_square_on_square():
    # When crop == image size, even tiny rotation shows border fill → 0 degrees
    angle = TextualInversionDataset._compute_max_rotation_angle(512, 512, 512, 512, 30)
    assert angle == 0.0


def test_compute_max_rotation_angle_wide_crop_returns_zero():
    # Crop wider than image: no safe rotation
    angle = TextualInversionDataset._compute_max_rotation_angle(100, 512, 200, 512, 30)
    assert angle == 0.0


def test_compute_max_rotation_angle_tall_crop_returns_zero():
    angle = TextualInversionDataset._compute_max_rotation_angle(512, 100, 512, 200, 30)
    assert angle == 0.0


def test_compute_max_rotation_angle_capped_at_max():
    # Even if math allows more, result is capped at max_degrees
    angle = TextualInversionDataset._compute_max_rotation_angle(1024, 1024, 512, 512, 30)
    assert angle == 30.0


def test_compute_max_rotation_angle_returns_non_negative():
    # Narrow image with moderate crop still returns non-negative
    angle = TextualInversionDataset._compute_max_rotation_angle(512, 512, 400, 400, 30)
    assert angle >= 0.0


def test_rotation_enabled_output_shape(sample_images, mock_images_dir):
    ds = TextualInversionDataset(
        sample_images, _FakeTokenizer(), "<sks>", size=512, repeats=1,
        augment_max_rotation_degrees=30,
    )
    item = ds[0]
    assert item["pixel_values"].shape == (3, 512, 512)


def test_rotation_disabled_output_shape(sample_images, mock_images_dir):
    ds = TextualInversionDataset(
        sample_images, _FakeTokenizer(), "<sks>", size=512, repeats=1,
        augment_max_rotation_degrees=0,
    )
    item = ds[0]
    assert item["pixel_values"].shape == (3, 512, 512)
