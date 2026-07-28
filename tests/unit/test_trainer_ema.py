"""Unit tests for EmbeddingEMA in backend/training/trainer.py"""
import torch
import torch.nn as nn

from backend.training.trainer import EmbeddingEMA


def _make_offsets(values: list[float]) -> nn.Parameter:
    return nn.Parameter(torch.tensor(values, dtype=torch.float32).unsqueeze(1))


def test_shadow_starts_as_clone_of_offsets():
    offsets = _make_offsets([1.0, 2.0, 3.0])
    ema = EmbeddingEMA(offsets, decay=0.9999)
    assert torch.equal(ema.shadow, offsets.data)


def test_update_applies_ema_formula():
    offsets = _make_offsets([1.0])
    ema = EmbeddingEMA(offsets, decay=0.9)
    # After one update with new offsets of 2.0:
    # shadow = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    offsets.data.fill_(2.0)
    ema.update(offsets)
    expected = torch.tensor([[1.1]])
    assert torch.allclose(ema.shadow, expected, atol=1e-6)


def test_get_returns_clone():
    offsets = _make_offsets([5.0])
    ema = EmbeddingEMA(offsets, decay=0.9999)
    result = ema.get()
    # Mutating the returned tensor should not affect shadow
    result.fill_(0.0)
    assert ema.shadow.item() != 0.0


def test_slow_convergence_at_high_decay():
    offsets = _make_offsets([0.0])
    ema = EmbeddingEMA(offsets, decay=0.9999)
    offsets.data.fill_(1.0)
    ema.update(offsets)
    # shadow = 0.9999 * 0 + 0.0001 * 1 = 0.0001
    expected = 0.0001
    assert abs(ema.shadow.item() - expected) < 1e-6


def test_multiple_updates():
    offsets = _make_offsets([0.0])
    ema = EmbeddingEMA(offsets, decay=0.5)
    offsets.data.fill_(1.0)
    ema.update(offsets)  # shadow = 0.5*0 + 0.5*1 = 0.5
    ema.update(offsets)  # shadow = 0.5*0.5 + 0.5*1 = 0.75
    assert abs(ema.shadow.item() - 0.75) < 1e-6
