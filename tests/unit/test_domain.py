"""Unit tests for backend/domain.py"""
import pytest
from backend.domain import ImageGenerationParams, QueueItem, ItemStatus, CriticalParams, make_params


def test_queue_item_gets_unique_ids():
    p = make_params()
    a = QueueItem(params=p)
    b = QueueItem(params=p)
    assert a.item_id != b.item_id


def test_queue_item_defaults_to_pending():
    item = QueueItem(params=make_params())
    assert item.status == ItemStatus.PENDING


def test_queue_item_output_image_name_defaults_to_none():
    item = QueueItem(params=make_params())
    assert item.output_image_name is None
    assert item.error is None


def test_critical_params_is_hashable():
    cp = CriticalParams(
        width=512, height=512, steps=20, cfg=7.5,
        cfg_rescale_multiplier=0.0, sampler="euler_a", model_key="k1"
    )
    # Must be usable as a dict key
    d = {cp: "value"}
    assert d[cp] == "value"


def test_critical_params_equality():
    cp1 = CriticalParams(width=512, height=512, steps=20, cfg=7.5,
                         cfg_rescale_multiplier=0.0, sampler="euler_a", model_key="k1")
    cp2 = CriticalParams(width=512, height=512, steps=20, cfg=7.5,
                         cfg_rescale_multiplier=0.0, sampler="euler_a", model_key="k1")
    assert cp1 == cp2


def test_make_params_returns_valid_params():
    p = make_params(prompt="hello")
    assert p.prompt == "hello"
    assert p.width == 512
    assert p.sampler == "euler_a"

