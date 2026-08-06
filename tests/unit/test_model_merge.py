"""Unit tests for model merge: definition-time validation + inference-time blend."""
import logging
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.domain import ModelEntry
from backend.model_registry.registry import ModelMergeError
from backend.model_registry.loader import (
    PipelineLoader,
    PipelineLoadError,
    _merge_transformer_state_dicts,
    _resolve_transformer_sd_paths,
    _load_merge_pipeline,
    validate_merge_components,
)
from tests.conftest import make_fake_sana_dir


def make_local_entry(key: str, name: str, path: Path) -> ModelEntry:
    return ModelEntry(key=key, name=name, base="sana",
                      source=str(path), source_type="local_path")


def test_validate_merge_components_passes(tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    d2 = make_fake_sana_dir(tmp_path, "b", ["a", "b"])
    e1 = make_local_entry("k1", "A", d1)
    e2 = make_local_entry("k2", "B", d2)
    validate_merge_components([e1, e2])  # should not raise


def test_validate_merge_components_mismatch(tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    d2 = make_fake_sana_dir(tmp_path, "b", ["a", "b", "c"])
    e1 = make_local_entry("k1", "A", d1)
    e2 = make_local_entry("k2", "B", d2)
    with pytest.raises(ModelMergeError) as excinfo:
        validate_merge_components([e1, e2])
    incompatible = excinfo.value.incompatible
    assert incompatible is not None
    assert "k2" in incompatible
    assert "k1" not in incompatible
    assert "different architecture" in incompatible["k2"]


def test_merge_weighted_sum(tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    d2 = make_fake_sana_dir(tmp_path, "b", ["a", "b"])
    e1 = make_local_entry("k1", "A", d1)
    e2 = make_local_entry("k2", "B", d2)
    merged = _merge_transformer_state_dicts([e1, e2], [1.0, 3.0])
    assert set(merged.keys()) == {"a", "b"}

    from safetensors import safe_open
    with safe_open(str(d1 / "transformer" / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as h:
        t1 = h.get_tensor("a")
    with safe_open(str(d2 / "transformer" / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as h:
        t2 = h.get_tensor("a")
    expected = t1 * 0.25 + t2 * 0.75
    assert torch.allclose(merged["a"], expected)


def test_merge_accumulates_in_native_dtype(tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a"], dtype=torch.float16)
    d2 = make_fake_sana_dir(tmp_path, "b", ["a"], dtype=torch.float16)
    e1 = make_local_entry("k1", "A", d1)
    e2 = make_local_entry("k2", "B", d2)
    merged = _merge_transformer_state_dicts([e1, e2], [1.0, 1.0])
    assert merged["a"].dtype == torch.float16


def test_merge_progressive_three_equal(tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    d2 = make_fake_sana_dir(tmp_path, "b", ["a", "b"])
    d3 = make_fake_sana_dir(tmp_path, "c", ["a", "b"])
    e1 = make_local_entry("k1", "A", d1)
    e2 = make_local_entry("k2", "B", d2)
    e3 = make_local_entry("k3", "C", d3)
    merged = _merge_transformer_state_dicts([e1, e2, e3], [1.0, 1.0, 1.0])
    assert set(merged.keys()) == {"a", "b"}

    from safetensors import safe_open
    with safe_open(str(d1 / "transformer" / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as h:
        t1 = h.get_tensor("a")
    with safe_open(str(d2 / "transformer" / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as h:
        t2 = h.get_tensor("a")
    with safe_open(str(d3 / "transformer" / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as h:
        t3 = h.get_tensor("a")
    expected = (t1 + t2 + t3) / 3
    assert torch.allclose(merged["a"], expected)


def test_merge_progressive_asymmetric(tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a"])
    d2 = make_fake_sana_dir(tmp_path, "b", ["a"])
    d3 = make_fake_sana_dir(tmp_path, "c", ["a"])
    d4 = make_fake_sana_dir(tmp_path, "d", ["a"])
    e1 = make_local_entry("k1", "A", d1)
    e2 = make_local_entry("k2", "B", d2)
    e3 = make_local_entry("k3", "C", d3)
    e4 = make_local_entry("k4", "D", d4)
    merged = _merge_transformer_state_dicts([e1, e2, e3, e4], [1.0, 1.0, 2.0, 1.0])

    from safetensors import safe_open
    with safe_open(str(d1 / "transformer" / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as h:
        t1 = h.get_tensor("a")
    with safe_open(str(d2 / "transformer" / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as h:
        t2 = h.get_tensor("a")
    with safe_open(str(d3 / "transformer" / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as h:
        t3 = h.get_tensor("a")
    with safe_open(str(d4 / "transformer" / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as h:
        t4 = h.get_tensor("a")
    expected = t1 / 5 + t2 / 5 + 2 * t3 / 5 + t4 / 5
    assert torch.allclose(merged["a"], expected)


def test_resolve_transformer_sd_paths_single(tmp_path):
    d = make_fake_sana_dir(tmp_path, "a", ["a"])
    e = make_local_entry("k1", "A", d)
    paths = _resolve_transformer_sd_paths(e)
    assert paths == [d / "transformer" / "diffusion_pytorch_model.safetensors"]


def test_resolve_transformer_sd_paths_sharded(tmp_path):
    tdir = tmp_path / "a" / "transformer"
    tdir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({"a": torch.randn(2, 4)}, str(tdir / "diffusion_pytorch_model-00001-of-00002.safetensors"))
    save_file({"b": torch.randn(2, 4)}, str(tdir / "diffusion_pytorch_model-00002-of-00002.safetensors"))
    e = make_local_entry("k1", "A", tmp_path / "a")
    paths = _resolve_transformer_sd_paths(e)
    assert paths == [
        tdir / "diffusion_pytorch_model-00001-of-00002.safetensors",
        tdir / "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]


def test_resolve_transformer_sd_paths_missing_raises(tmp_path):
    e = make_local_entry("k1", "A", tmp_path / "empty")
    with pytest.raises(PipelineLoadError):
        _resolve_transformer_sd_paths(e)


def test_loader_load_merge_pipeline(tmp_path, monkeypatch):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    d2 = make_fake_sana_dir(tmp_path, "b", ["a", "b"])
    from backend.model_registry.registry import ModelRegistry
    registry = ModelRegistry(tmp_path / "models.json")
    registry.add(make_local_entry("k1", "A", d1))
    registry.add(make_local_entry("k2", "B", d2))
    merge_entry = registry.add_merge("Blend", [("k1", 1.0), ("k2", 1.0)])

    load_state_dict = MagicMock()
    fake_pipeline = SimpleNamespace(transformer=SimpleNamespace(load_state_dict=load_state_dict))
    monkeypatch.setattr(
        "backend.model_registry.loader._load_sana_pipeline",
        lambda entry: fake_pipeline,
    )

    loader = PipelineLoader(registry)
    pipeline = loader.load(merge_entry)
    assert pipeline is fake_pipeline
    assert load_state_dict.call_count == 1
    assert loader.loaded_key == merge_entry.key


def test_loader_merge_missing_component_raises(tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    from backend.model_registry.registry import ModelRegistry
    registry = ModelRegistry(tmp_path / "models.json")
    registry.add(make_local_entry("k1", "A", d1))
    registry.add(make_local_entry("k2", "B", d1))
    merge_entry = registry.add_merge("Blend", [("k1", 1.0), ("k2", 1.0)])
    registry.delete("k2")

    loader = PipelineLoader(registry)
    with pytest.raises(PipelineLoadError, match="no longer in the registry"):
        _load_merge_pipeline(merge_entry, registry)


def test_loader_without_registry_raises(tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a"])
    from backend.model_registry.registry import ModelRegistry
    registry = ModelRegistry(tmp_path / "models.json")
    registry.add(make_local_entry("k1", "A", d1))
    registry.add(make_local_entry("k2", "B", d1))
    merge_entry = registry.add_merge("Blend", [("k1", 1.0), ("k2", 1.0)])
    loader = PipelineLoader(None)
    with pytest.raises(PipelineLoadError, match="without a registry"):
        loader.load(merge_entry)


def test_bnb_8bit_cast_warning_is_suppressed():
    logger = logging.getLogger("bitsandbytes.autograd._functions")
    emitted = []
    handler = logging.Handler()
    handler.emit = lambda record: emitted.append(record.getMessage())
    logger.addHandler(handler)
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        logger.warning(
            "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
        )
        logger.warning("some unrelated warning")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    assert emitted == ["some unrelated warning"]
