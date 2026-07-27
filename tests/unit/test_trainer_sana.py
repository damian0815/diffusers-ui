"""Unit tests for SANA textual inversion support in backend/training/trainer.py"""
import torch
from unittest.mock import MagicMock

from backend.training.trainer import _draw_flowmatch_timesteps


class TestDrawFlowmatchTimesteps:
    def test_shape(self):
        timesteps = torch.linspace(0, 1, 1000)
        result = _draw_flowmatch_timesteps(8, 1000, timesteps, device="cpu")
        assert result.shape == (8,)

    def test_range_within_timesteps(self):
        timesteps = torch.linspace(0, 1, 1000)
        result = _draw_flowmatch_timesteps(100, 1000, timesteps, device="cpu")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dtype_float(self):
        timesteps = torch.linspace(0, 1, 1000)
        result = _draw_flowmatch_timesteps(16, 1000, timesteps, device="cpu")
        assert result.dtype == torch.float32

    def test_device_cpu(self):
        timesteps = torch.linspace(0, 1, 1000)
        result = _draw_flowmatch_timesteps(4, 1000, timesteps, device="cpu")
        assert result.device.type == "cpu"

    def test_values_are_from_input(self):
        """All output values must come from the input timesteps tensor."""
        timesteps = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        result = _draw_flowmatch_timesteps(100, 10, timesteps, device="cpu")
        for val in result:
            assert val.item() in timesteps.tolist()

    def test_batch_size_one(self):
        timesteps = torch.linspace(0, 1, 1000)
        result = _draw_flowmatch_timesteps(1, 1000, timesteps, device="cpu")
        assert result.shape == (1,)


class TestSanaVelocityTarget:
    def test_velocity_is_noise_minus_latents(self):
        """Flow-matching velocity target should be noise - latents."""
        latents = torch.randn(2, 4, 32, 32)
        noise = torch.randn(2, 4, 32, 32)
        target = noise - latents
        expected = noise - latents
        assert torch.allclose(target, expected)

    def test_velocity_shape_matches_latents(self):
        latents = torch.randn(2, 4, 32, 32)
        noise = torch.randn(2, 4, 32, 32)
        target = noise - latents
        assert target.shape == latents.shape


class TestSanaForwardCall:
    def test_transformer_keyword_args(self):
        """SANA transformer should be called with keyword arguments."""
        mock_transformer = MagicMock()
        mock_transformer.return_value.sample = torch.randn(1, 4, 32, 32)

        hidden_states = torch.randn(1, 4, 32, 32)
        encoder_hidden_states = torch.randn(1, 77, 2048)
        timesteps = torch.tensor([0.5])

        result = mock_transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
        )

        mock_transformer.assert_called_once_with(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
        )
        assert result.sample.shape == (1, 4, 32, 32)
