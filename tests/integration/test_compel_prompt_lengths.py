"""Integration tests: build_prompt_embeds with real CompelForSD.

Uses a tiny in-memory tokenizer/text encoder so no model download or GPU is
needed. This covers the prompt-length handling that motivated switching SD1/SD2
to the CompelForSD convenience wrapper: positives and negatives must be padded
to a common sequence length, and long prompts must be chunked.
"""
from types import SimpleNamespace

import pytest
import torch

from backend.pipeline.compel_encoder import build_prompt_embeds


class FakeTokenizer:
    """Minimal tokenizer: one token per whitespace-separated word."""

    model_max_length = 12
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, texts, truncation, padding, return_tensors):
        return {
            "input_ids": [
                [self.bos_token_id]
                + [3 + (i % 5) for i, _ in enumerate(text.split())]
                + [self.eos_token_id]
                for text in texts
            ]
        }

    def convert_ids_to_tokens(self, ids):
        return [f"{token_id}</w>" for token_id in ids]

    def tokenize(self, text):
        return text.split()


class FakeTextEncoder:
    """Turns token IDs into trivially shaped hidden states."""

    device = "cpu"
    dtype = torch.float32

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True):
        ids = input_ids.long()
        hidden = torch.zeros((*ids.shape, 4), dtype=torch.float32)
        hidden[..., 0] = ids
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=[hidden, hidden])


@pytest.fixture
def fake_sd_pipeline():
    return SimpleNamespace(tokenizer=FakeTokenizer(), text_encoder=FakeTextEncoder())


def test_build_prompt_embeds_pads_different_prompt_lengths_together(fake_sd_pipeline):
    kwargs = build_prompt_embeds(
        fake_sd_pipeline,
        ["one two", "one two three four five"],
        ["bad", "bad bad bad"],
        is_sdxl=False,
    )

    assert kwargs["prompt_embeds"].shape == torch.Size([2, 12, 4])
    assert kwargs["negative_prompt_embeds"].shape == kwargs["prompt_embeds"].shape
    assert "prompt" not in kwargs
    assert "negative_prompt" not in kwargs


def test_build_prompt_embeds_chunks_long_prompts(fake_sd_pipeline):
    long_prompt = " ".join(f"word{i}" for i in range(20))

    kwargs = build_prompt_embeds(
        fake_sd_pipeline,
        [long_prompt, "one two"],
        ["bad"],
        is_sdxl=False,
    )

    # Two 12-token chunks for the long prompt; everything padded to 24 tokens.
    assert kwargs["prompt_embeds"].shape == torch.Size([2, 24, 4])
    assert kwargs["negative_prompt_embeds"].shape == kwargs["prompt_embeds"].shape
