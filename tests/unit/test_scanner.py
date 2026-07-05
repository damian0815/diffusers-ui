"""Unit tests for backend/model_registry/scanner.py"""
import json
import pytest
from pathlib import Path
from backend.model_registry.scanner import (
    scan_local_dir,
    _looks_like_pipeline,
    _detect_base,
    _guess_base_from_name,
)


def _write_index(pipeline_dir, **components):
    """Write model_index.json with the given component -> [lib, class] entries."""
    (pipeline_dir / "model_index.json").write_text(json.dumps(components))


def _write_unet_config(pipeline_dir, **kwargs):
    d = pipeline_dir / "unet"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(kwargs))


def test_scan_local_dir_finds_pipeline(tmp_path):
    pipeline_dir = tmp_path / "my_model"
    pipeline_dir.mkdir()
    _write_index(pipeline_dir, vae=["diffusers", "AutoencoderKL"])
    results = scan_local_dir(tmp_path)
    assert any(r.source == str(pipeline_dir) for r in results)


def test_scan_local_dir_ignores_non_pipelines(tmp_path):
    (tmp_path / "random_folder").mkdir()
    results = scan_local_dir(tmp_path)
    assert results == []


def test_scan_local_dir_finds_multiple_pipelines(tmp_path):
    for name in ["model_a", "model_b"]:
        d = tmp_path / name
        d.mkdir()
        _write_index(d, vae=["diffusers", "AutoencoderKL"])
    results = scan_local_dir(tmp_path)
    assert len(results) == 2


def test_scan_local_dir_sets_correct_source(tmp_path):
    d = tmp_path / "mymodel"
    d.mkdir()
    _write_index(d, vae=["diffusers", "AutoencoderKL"])
    results = scan_local_dir(tmp_path)
    assert results[0].source_type == "local_path"
    assert results[0].source == str(d)


def test_scan_local_dir_nonexistent_dir_returns_empty():
    results = scan_local_dir(Path("/nonexistent/path/xyz"))
    assert results == []


def test_looks_like_pipeline_with_model_index(tmp_path):
    _write_index(tmp_path, vae=["diffusers", "AutoencoderKL"])
    assert _looks_like_pipeline(tmp_path) is True


def test_looks_like_pipeline_without_model_index(tmp_path):
    assert _looks_like_pipeline(tmp_path) is False


# ── _detect_base ─────────────────────────────────────────────────────


def test_detect_base_sana_via_transformer(tmp_path):
    _write_index(tmp_path, transformer=["diffusers", "SanaTransformer2DModel"])
    assert _detect_base(tmp_path) == "sana"


def test_detect_base_sdxl_via_text_encoder_2(tmp_path):
    _write_index(tmp_path, unet=["diffusers", "UNet2DConditionModel"],
                 text_encoder_2=["transformers", "CLIPTextModelWithProjection"])
    assert _detect_base(tmp_path) == "sdxl"


def test_detect_base_sd1_default(tmp_path):
    _write_index(tmp_path, unet=["diffusers", "UNet2DConditionModel"])
    assert _detect_base(tmp_path) == "sd-1"


def test_detect_base_sd2_via_cross_attention_dim(tmp_path):
    _write_index(tmp_path, unet=["diffusers", "UNet2DConditionModel"])
    _write_unet_config(tmp_path, cross_attention_dim=1024)
    assert _detect_base(tmp_path) == "sd-2"


def test_detect_base_sd1_with_768_cross_attention(tmp_path):
    _write_index(tmp_path, unet=["diffusers", "UNet2DConditionModel"])
    _write_unet_config(tmp_path, cross_attention_dim=768)
    assert _detect_base(tmp_path) == "sd-1"


def test_detect_base_unknown_index_falls_back_to_name(tmp_path):
    """Missing model_index.json should fall back to guessing from directory name."""
    d = tmp_path / "sana-flash"
    d.mkdir()
    assert _detect_base(d) == "sana"


# ── _guess_base_from_name (fallback heuristic) ───────────────────────


def test_guess_base_sdxl():
    assert _guess_base_from_name("stable-diffusion-xl-base") == "sdxl"
    assert _guess_base_from_name("SDXL-turbo") == "sdxl"


def test_guess_base_sd2():
    assert _guess_base_from_name("stable-diffusion-2.0") == "sd-2"
    assert _guess_base_from_name("sd2-base") == "sd-2"


def test_guess_base_sd1_default():
    assert _guess_base_from_name("some-random-model") == "sd-1"
    assert _guess_base_from_name("stable-diffusion-v1-5") == "sd-1"

