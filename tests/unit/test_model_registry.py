"""Unit tests for backend/model_registry/registry.py"""
import pytest
from pathlib import Path
from backend.model_registry.registry import ModelRegistry, ModelNotFoundError
from backend.domain import ModelEntry


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "models.json")


def make_entry(key="k1", name="My Model", base="sd-2",
               source="org/repo", source_type="hf_repo_id") -> ModelEntry:
    return ModelEntry(key=key, name=name, base=base,
                      source=source, source_type=source_type)


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

