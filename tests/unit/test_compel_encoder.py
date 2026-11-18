"""Unit tests for backend/pipeline/compel_encoder.py."""
import compel
import pytest

from backend.pipeline.compel_encoder import build_prompt_embeds, build_sana_prompt_embeds


class FakeConditioning:
    """Shaped like compel's LabelledConditioning."""

    def __init__(
        self,
        embeds,
        negative_embeds,
        attention_mask=None,
        negative_attention_mask=None,
        pooled_embeds=None,
        negative_pooled_embeds=None,
    ):
        self.embeds = embeds
        self.negative_embeds = negative_embeds
        self.attention_mask = attention_mask
        self.negative_attention_mask = negative_attention_mask
        self.pooled_embeds = pooled_embeds
        self.negative_pooled_embeds = negative_pooled_embeds


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


@pytest.fixture
def fake_prompt_wrappers(monkeypatch):
    """Replace CompelForSD / CompelForSDXL with stubs that record their args."""
    state = {"sd": None, "sdxl": None}

    class FakeWrapper:
        def __init__(self, kind, pipe):
            self.kind = kind
            self.pipe = pipe

        def __call__(self, prompts, negative_prompt=None):
            state[self.kind] = {
                "pipeline": self.pipe,
                "prompts": prompts,
                "negative_prompt": negative_prompt,
            }
            return FakeConditioning(
                embeds=f"{self.kind}-embeds",
                negative_embeds=f"{self.kind}-neg-embeds",
                pooled_embeds=f"{self.kind}-pooled",
                negative_pooled_embeds=f"{self.kind}-neg-pooled",
            )

    class FakeCompelForSD(FakeWrapper):
        def __init__(self, pipe):
            super().__init__("sd", pipe)

    class FakeCompelForSDXL(FakeWrapper):
        def __init__(self, pipe):
            super().__init__("sdxl", pipe)

    monkeypatch.setattr(compel, "CompelForSD", FakeCompelForSD)
    monkeypatch.setattr(compel, "CompelForSDXL", FakeCompelForSDXL)
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


def test_build_prompt_embeds_sd1_and_sd2_use_compel_for_sd(fake_prompt_wrappers):
    """SD 1.x and 2.x share a single text encoder, encoded by CompelForSD."""
    prompts = ["a cat", "a very long description of a dog"]
    neg_prompts = ["blur", "noise"]

    kwargs = build_prompt_embeds(
        "fake-pipeline", prompts, neg_prompts, is_sdxl=False
    )

    assert kwargs == {
        "prompt_embeds": "sd-embeds",
        "negative_prompt_embeds": "sd-neg-embeds",
    }
    assert fake_prompt_wrappers["sd"] == {
        "pipeline": "fake-pipeline",
        "prompts": prompts,
        "negative_prompt": neg_prompts,
    }
    assert fake_prompt_wrappers["sdxl"] is None


def test_build_prompt_embeds_sdxl_uses_compel_for_sdxl_with_pooled(fake_prompt_wrappers):
    prompts = ["a cat", "a dog"]
    neg_prompts = ["blur", "noise"]

    kwargs = build_prompt_embeds(
        "fake-pipeline", prompts, neg_prompts, is_sdxl=True
    )

    assert kwargs == {
        "prompt_embeds": "sdxl-embeds",
        "negative_prompt_embeds": "sdxl-neg-embeds",
        "pooled_prompt_embeds": "sdxl-pooled",
        "negative_pooled_prompt_embeds": "sdxl-neg-pooled",
    }
    assert fake_prompt_wrappers["sdxl"]["prompts"] == prompts
    assert fake_prompt_wrappers["sdxl"]["negative_prompt"] == neg_prompts
    assert fake_prompt_wrappers["sd"] is None
