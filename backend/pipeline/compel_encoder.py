"""
Compel-based prompt encoding for SD 1.x / 2.x, SDXL and SANA pipelines.

Compel (https://github.com/damian0815/compel) enables:
  - Weighted prompt syntax: (term:1.5), (term:0.8), (term--)
  - Prompts longer than 77 tokens via chunked encoding
    (requires truncate_long_prompts=False)
  - Blending / conjunction of multiple prompts

Usage (SD 1.x / 2.x):
    kwargs = build_prompt_embeds(pipeline, prompts, neg_prompts, is_sdxl=False)
    pipeline(**kwargs, width=..., ...)

Usage (SDXL):
    kwargs = build_prompt_embeds(pipeline, prompts, neg_prompts, is_sdxl=True)
    pipeline(**kwargs, width=..., ...)

Usage (SANA):
    kwargs = build_sana_prompt_embeds(pipeline, prompts, neg_prompts)
    pipeline(**kwargs, width=..., ...)
"""
from __future__ import annotations


def build_prompt_embeds(
    pipeline,
    prompts: list[str],
    neg_prompts: list[str],
    *,
    is_sdxl: bool,
) -> dict:
    """
    Encode *prompts* and *neg_prompts* with the matching CompelFor* wrapper
    and return a kwargs dict ready to be unpacked into the diffusers pipeline.

    For SD 1.x / 2.x the dict contains:
        prompt_embeds, negative_prompt_embeds

    For SDXL the dict additionally contains:
        pooled_prompt_embeds, negative_pooled_prompt_embeds

    The *prompt* / *negative_prompt* string keys are intentionally absent so
    that diffusers uses the pre-computed embeddings path and does not re-encode.

    Both wrappers encode positives and negatives in a *single* Compel call.
    With truncate_long_prompts=False that call pads every conditioning to a
    common sequence length; encoding the two lists separately (the old SD1/SD2
    approach) lets their sequence lengths diverge when prompts have different
    lengths, which the pipeline cannot accept as a single batch.

    Args:
        pipeline:    a loaded diffusers pipeline with tokenizer(s) and
                     text_encoder(s) available.
        prompts:     positive prompt strings, one per batch item.
        neg_prompts: negative prompt strings, same length as *prompts*.
        is_sdxl:     True for SDXL (dual-encoder), False for SD 1.x / 2.x.

    Returns:
        dict of pipeline keyword arguments.
    """
    from compel import CompelForSD, CompelForSDXL

    wrapper_cls = CompelForSDXL if is_sdxl else CompelForSD
    conditioning = wrapper_cls(pipeline)(prompts, negative_prompt=neg_prompts)

    kwargs = {
        "prompt_embeds": conditioning.embeds,
        "negative_prompt_embeds": conditioning.negative_embeds,
    }
    if is_sdxl:
        kwargs["pooled_prompt_embeds"] = conditioning.pooled_embeds
        kwargs["negative_pooled_prompt_embeds"] = conditioning.negative_pooled_embeds
    return kwargs


def build_sana_prompt_embeds(
    pipeline,
    prompts: list[str],
    neg_prompts: list[str],
) -> dict:
    """
    Encode *prompts* and *neg_prompts* for a SANA pipeline using
    ``CompelForSana`` (which supports Gemma, SANA's text encoder).

    ``CompelForSana`` returns a ``LabelledConditioning`` whose
    ``attention_mask`` / ``negative_attention_mask`` map to the SanaPipeline
    kwargs *prompt_attention_mask* / *negative_prompt_attention_mask* (note
    the ``prompt_`` prefix).

    ``prompt`` / ``negative_prompt`` are forwarded as ``None`` so the
    pre-computed embeddings path is used: ``SanaPAGPipeline.__call__`` defaults
    ``negative_prompt`` to ``""`` (not ``None``) and would otherwise trip its
    "cannot forward both negative_prompt and negative_prompt_embeds" guard.

    Args:
        pipeline:    a loaded ``SanaPipeline`` (or ``SanaPAGPipeline``).
        prompts:     positive prompt strings, one per batch item.
        neg_prompts: negative prompt strings, same length as *prompts*.

    Returns:
        dict of pipeline keyword arguments.
    """
    from compel import CompelForSana

    cond = CompelForSana(pipeline)(prompts, negative_prompt=neg_prompts)
    return {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": cond.embeds,
        "negative_prompt_embeds": cond.negative_embeds,
        "prompt_attention_mask": cond.attention_mask,
        "negative_prompt_attention_mask": cond.negative_attention_mask,
    }
