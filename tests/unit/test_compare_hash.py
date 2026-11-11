"""Unit tests for backend/gallery/compare.py — canonical hashing (plan §6.1).

compare.py only does getattr/mapping reads, so stub objects and plain dicts
stand in for ImageRecord — no ORM, no DB.
"""
from types import SimpleNamespace

import pytest

from backend.gallery.compare import (
    ALL_FIELDS,
    FIELD_COLUMNS,
    canonical_payload,
    comparison_hash,
    validation_error_fields,
)

DEFAULT_FIELDS = ["prompt", "negative_prompt", "seed", "width", "height", "cfg"]


def _record(**overrides):
    """A stub image record with plausible generation values."""
    values = dict(
        prompt="a cat",
        negative_prompt="blurry",
        seed=42,
        width=512,
        height=512,
        cfg=7.5,
        model_key="uuid-a",
        steps=20,
        sampler="euler_a",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_hash_independent_of_dict_build_order():
    a = _record()
    b = _record()
    forward = comparison_hash(a, DEFAULT_FIELDS)
    reversed_ = comparison_hash(b, list(reversed(DEFAULT_FIELDS)))
    assert forward == reversed_


def test_hash_independent_of_dict_key_order():
    """Dict-backed records hash the same regardless of insertion order."""
    d1 = {"prompt": "a cat", "seed": 42}
    d2 = {"seed": 42, "prompt": "a cat"}
    assert comparison_hash(d1, ["prompt", "seed"]) == comparison_hash(d2, ["seed", "prompt"])


def test_non_metric_field_differences_are_ignored():
    a = _record(steps=20, sampler="euler_a")
    b = _record(steps=50, sampler="dpmpp_2m")
    assert comparison_hash(a, DEFAULT_FIELDS) == comparison_hash(b, DEFAULT_FIELDS)


def test_hash_null_normalization():
    """NULL vs default must hash equal (§3.4)."""
    assert comparison_hash(_record(prompt=None), ["prompt"]) == comparison_hash(
        _record(prompt=""), ["prompt"]
    )
    assert comparison_hash(_record(negative_prompt=None), ["negative_prompt"]) == comparison_hash(
        _record(negative_prompt=""), ["negative_prompt"]
    )
    assert comparison_hash(_record(model_key=None), ["model"]) == comparison_hash(
        _record(model_key=""), ["model"]
    )
    assert comparison_hash(_record(seed=None), ["seed"]) == comparison_hash(
        _record(seed=0), ["seed"]
    )
    assert comparison_hash(_record(width=None), ["width"]) == comparison_hash(
        _record(width=0), ["width"]
    )
    assert comparison_hash(_record(height=None), ["height"]) == comparison_hash(
        _record(height=0), ["height"]
    )
    assert comparison_hash(_record(cfg=None), ["cfg"]) == comparison_hash(
        _record(cfg=0.0), ["cfg"]
    )


def test_hash_differs_when_metric_field_differs():
    base = comparison_hash(_record(), DEFAULT_FIELDS)
    assert comparison_hash(_record(seed=43), DEFAULT_FIELDS) != base
    assert comparison_hash(_record(prompt="a dog"), DEFAULT_FIELDS) != base


def test_hash_empty_fields_is_constant():
    """fields=[] ⇒ every record hashes the same (§3.4, deliberate)."""
    a = _record(prompt="a cat", seed=1)
    b = _record(prompt="a dog", seed=2, model_key="uuid-b")
    assert comparison_hash(a, []) == comparison_hash(b, [])


def test_unknown_field_raises():
    with pytest.raises(ValueError, match="unknown comparison fields"):
        comparison_hash(_record(), ["steps"])
    with pytest.raises(ValueError):
        canonical_payload(_record(), ["sampler"])


def test_validation_error_fields_lists_all_unknown_sorted():
    assert validation_error_fields(["seed", "steps", "zzz", "foo"]) == ["foo", "steps", "zzz"]
    assert validation_error_fields(ALL_FIELDS) == []


def test_all_fields_matches_field_columns():
    assert ALL_FIELDS == sorted(FIELD_COLUMNS)
    # The `model` field maps to the stable model_key, not the name snapshot (§3.3).
    assert FIELD_COLUMNS["model"][0] == "model_key"
