"""Unit tests for backend/pipeline/batcher.py"""
import pytest
from backend.pipeline.batcher import group_into_batches, get_critical_params
from backend.domain import QueueItem, make_params


def make_item(**overrides):
    return QueueItem(params=make_params(**overrides))


def test_empty_input():
    assert group_into_batches([]) == []


def test_same_critical_params_grouped_together():
    a, b = make_item(), make_item()
    groups = group_into_batches([a, b])
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_different_sizes_separate_groups():
    a = make_item(width=512)
    b = make_item(width=768)
    groups = group_into_batches([a, b])
    assert len(groups) == 2


def test_different_model_keys_separate_groups():
    a = make_item(model_key="model_a")
    b = make_item(model_key="model_b")
    groups = group_into_batches([a, b])
    assert len(groups) == 2


def test_different_samplers_separate_groups():
    a = make_item(sampler="euler_a")
    b = make_item(sampler="ddim")
    groups = group_into_batches([a, b])
    assert len(groups) == 2


def test_preserves_order_within_group():
    items = [make_item() for _ in range(5)]
    groups = group_into_batches(items)
    assert groups[0] == items


def test_three_different_models_three_groups():
    a = make_item(model_key="m1")
    b = make_item(model_key="m2")
    c = make_item(model_key="m1")
    groups = group_into_batches([a, b, c])
    assert len(groups) == 2
    # m1 group should contain both a and c
    m1_group = next(g for g in groups if g[0].params.model_key == "m1")
    assert len(m1_group) == 2
    assert m1_group[0].item_id == a.item_id
    assert m1_group[1].item_id == c.item_id


def test_get_critical_params_extracts_correctly():
    params = make_params(width=768, steps=30, cfg=9.0, model_key="xyz")
    cp = get_critical_params(params)
    assert cp.width == 768
    assert cp.steps == 30
    assert cp.cfg == 9.0
    assert cp.model_key == "xyz"

