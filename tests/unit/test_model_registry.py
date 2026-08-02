"""Unit tests for backend/model_registry/registry.py"""
import pytest
from pathlib import Path
from backend.model_registry.registry import ModelRegistry, ModelNotFoundError, ModelMergeError
from backend.domain import ModelEntry, MergeComponent


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "models.json")


def make_entry(key="k1", name="My Model", base="sd-2",
               source="org/repo", source_type="hf_repo_id") -> ModelEntry:
    return ModelEntry(key=key, name=name, base=base,
                      source=source, source_type=source_type)


def make_sana_entry(key, name, source="org/sana", source_type="hf_repo_id",
                    hf_api_key=None) -> ModelEntry:
    return ModelEntry(key=key, name=name, base="sana",
                      source=source, source_type=source_type,
                      hf_api_key=hf_api_key)


def test_empty_registry_returns_empty_list(registry):
    assert registry.all() == []


def test_add_and_retrieve(registry):
    entry = make_entry()
    registry.add(entry)
    assert registry.get("k1").name == "My Model"


def test_get_nonexistent_returns_none(registry):
    assert registry.get("nope") is None


def test_all_returns_all_entries(registry):
    registry.add(make_entry("k1", "M1"))
    registry.add(make_entry("k2", "M2"))
    names = {e.name for e in registry.all()}
    assert names == {"M1", "M2"}


def test_rename(registry):
    registry.add(make_entry())
    registry.rename("k1", "New Name")
    assert registry.get("k1").name == "New Name"


def test_rename_nonexistent_raises(registry):
    with pytest.raises(ModelNotFoundError):
        registry.rename("nope", "X")


def test_delete(registry):
    registry.add(make_entry())
    registry.delete("k1")
    assert registry.get("k1") is None


def test_delete_nonexistent_raises(registry):
    with pytest.raises(ModelNotFoundError):
        registry.delete("nope")


def test_persists_across_instances(tmp_path):
    path = tmp_path / "models.json"
    r1 = ModelRegistry(path)
    r1.add(ModelEntry(key="k1", name="Persist", base="sd-2",
                      source="org/repo", source_type="hf_repo_id"))
    r2 = ModelRegistry(path)
    assert r2.get("k1").name == "Persist"


def test_register_new_generates_uuid(tmp_path):
    r = ModelRegistry(tmp_path / "m.json")
    entry = r.register_new(name="Test", base="sd-1", source="a/b",
                           source_type="hf_repo_id")
    assert len(entry.key) == 36  # UUID format
    assert r.get(entry.key) is not None


# ---------------------------------------------------------------------------
# Merge model tests
# ---------------------------------------------------------------------------

def test_add_merge_creates_entry(registry):
    k1 = registry.add(make_sana_entry("k1", "Sana A", hf_api_key="secret"))
    k2 = registry.add(make_sana_entry("k2", "Sana B"))
    merged = registry.add_merge("Blend", [("k1", 1.0), ("k2", 2.0)])
    assert merged.base == "sana"
    assert merged.source == "org/sana"
    assert merged.source_type == "hf_repo_id"
    assert merged.hf_api_key == "secret"
    assert merged.quantization is None
    assert isinstance(merged.merge, list)
    assert len(merged.merge) == 2
    assert all(isinstance(c, MergeComponent) for c in merged.merge)
    assert [c.key for c in merged.merge] == ["k1", "k2"]
    assert [c.weight for c in merged.merge] == [1.0, 2.0]
    assert registry.get(merged.key) is merged


def test_merge_persists_across_instances(tmp_path):
    path = tmp_path / "models.json"
    r1 = ModelRegistry(path)
    r1.add(make_sana_entry("k1", "A"))
    r1.add(make_sana_entry("k2", "B"))
    merged = r1.add_merge("Blend", [("k1", 1.0), ("k2", 1.5)])
    r2 = ModelRegistry(path)
    restored = r2.get(merged.key)
    assert restored.merge is not None
    assert all(isinstance(c, MergeComponent) for c in restored.merge)
    assert restored.merge[1].weight == 1.5


def test_merge_requires_at_least_two(registry):
    registry.add(make_sana_entry("k1", "A"))
    with pytest.raises(ModelMergeError):
        registry.add_merge("Blend", [("k1", 1.0)])


def test_merge_rejects_missing_key(registry):
    registry.add(make_sana_entry("k1", "A"))
    with pytest.raises(ModelMergeError, match="No model with key"):
        registry.add_merge("Blend", [("k1", 1.0), ("nope", 1.0)])


def test_merge_rejects_non_sana(registry):
    registry.add(make_sana_entry("k1", "A"))
    registry.add(make_entry("k2", "SD", base="sd-1"))
    with pytest.raises(ModelMergeError, match="only SANA"):
        registry.add_merge("Blend", [("k1", 1.0), ("k2", 1.0)])


def test_merge_rejects_nested(registry):
    k1 = registry.add(make_sana_entry("k1", "A"))
    k2 = registry.add(make_sana_entry("k2", "B"))
    merged = registry.add_merge("Blend", [("k1", 1.0), ("k2", 1.0)])
    with pytest.raises(ModelMergeError, match="nested"):
        registry.add_merge("Nested", [(merged.key, 1.0), ("k1", 1.0)])


@pytest.mark.parametrize("bad_weight", [0, -1])
def test_merge_rejects_non_positive_weight(registry, bad_weight):
    registry.add(make_sana_entry("k1", "A"))
    registry.add(make_sana_entry("k2", "B"))
    with pytest.raises(ModelMergeError, match="positive"):
        registry.add_merge("Blend", [("k1", 1.0), ("k2", bad_weight)])

