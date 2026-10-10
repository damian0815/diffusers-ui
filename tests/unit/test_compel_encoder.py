"""Unit tests for backend/pipeline/compel_encoder.py — SANA prompt encoding."""
import types

import compel
import pytest

from backend.pipeline.compel_encoder import build_sana_prompt_embeds


class FakeConditioning:
    """Shaped like compel's LabelledConditioning."""

    def __init__(self, embeds, negative_embeds, attention_mask, negative_attention_mask):
        self.embeds = embeds
        self.negative_embeds = negative_embeds
        self.attention_mask = attention_mask
        self.negative_attention_mask = negative_attention_mask


@pytest.fixture
def fake_compel(monkeypatch):
    """Replace compel.CompelForSana with a stub that records its call args."""
    state = {"prompts": None, "neg_prompts": None}

    class FakeCompelForSana:
        def __init__(self, pipe):
            self.pipe = pipe

        def __call__(self, prompts, negative_prompt=None):
            state["prompts"] = prompts
            state["neg_prompts"] = negative_prompt
            return FakeConditioning(
                embeds="embeds-tensor",
                negative_embeds="neg-embeds-tensor",
                attention_mask="mask-tensor",
                negative_attention_mask="neg-mask-tensor",
            )

    monkeypatch.setattr(compel, "CompelForSana", FakeCompelForSana)
    return state


def test_build_sana_prompt_embeds_maps_conditioning_keys(fake_compel):
    kwargs = build_sana_prompt_embeds("fake-pipeline", ["a cat"], ["blur"])

    assert kwargs == {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": "embeds-tensor",
        "negative_prompt_embeds": "neg-embeds-tensor",
        "prompt_attention_mask": "mask-tensor",
        "negative_prompt_attention_mask": "neg-mask-tensor",
    }


def test_build_sana_prompt_embeds_forwards_prompt_strings_as_none(fake_compel):
    """SanaPAGPipeline.__call__ defaults negative_prompt to '' — the embeds
    path must override it with None or diffusers raises
    "Cannot forward both negative_prompt and negative_prompt_embeds"."""
    kwargs = build_sana_prompt_embeds("fake-pipeline", ["a cat"], [""])

    assert kwargs["prompt"] is None
    assert kwargs["negative_prompt"] is None


def test_build_sana_prompt_embeds_passes_batched_lists(fake_compel):
    prompts = ["a cat", "a dog", "a bird"]
    neg_prompts = ["blur", "noise", "text"]

    build_sana_prompt_embeds("fake-pipeline", prompts, neg_prompts)

    assert fake_compel["prompts"] == prompts
    assert fake_compel["neg_prompts"] == neg_prompts
