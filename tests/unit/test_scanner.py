"""Unit tests for backend/model_registry/scanner.py"""
import pytest
from pathlib import Path
from backend.model_registry.scanner import scan_local_dir, _looks_like_pipeline, _guess_base


def test_scan_local_dir_finds_pipeline(tmp_path):
    pipeline_dir = tmp_path / "my_model"
    pipeline_dir.mkdir()
    (pipeline_dir / "model_index.json").write_text("{}")
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
        (d / "model_index.json").write_text("{}")
    results = scan_local_dir(tmp_path)
    assert len(results) == 2


def test_scan_local_dir_sets_correct_source(tmp_path):
    d = tmp_path / "mymodel"
    d.mkdir()
    (d / "model_index.json").write_text("{}")
    results = scan_local_dir(tmp_path)
    assert results[0].source_type == "local_path"
    assert results[0].source == str(d)


def test_scan_local_dir_nonexistent_dir_returns_empty():
    results = scan_local_dir(Path("/nonexistent/path/xyz"))
    assert results == []


def test_looks_like_pipeline_with_model_index(tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    assert _looks_like_pipeline(tmp_path) is True


def test_looks_like_pipeline_without_model_index(tmp_path):
    assert _looks_like_pipeline(tmp_path) is False


def test_guess_base_sdxl():
    assert _guess_base("stable-diffusion-xl-base") == "sdxl"
    assert _guess_base("SDXL-turbo") == "sdxl"


def test_guess_base_sd2():
    assert _guess_base("stable-diffusion-2.0") == "sd-2"
    assert _guess_base("sd2-base") == "sd-2"


def test_guess_base_sd1_default():
    assert _guess_base("some-random-model") == "sd-1"
    assert _guess_base("stable-diffusion-v1-5") == "sd-1"

