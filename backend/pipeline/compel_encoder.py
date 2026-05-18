"""
Compel-based prompt encoding for SD 1.x and SDXL pipelines.

Compel (https://github.com/damian0815/compel) enables:
  - Weighted prompt syntax: (term:1.5), (term:0.8), (term--)
  - Prompts longer than 77 tokens via chunked encoding
    (requires truncate_long_prompts=False)
  - Blending / conjunction of multiple prompts

Usage (SD 1.x):
    kwargs = build_prompt_embeds(pipeline, prompts, neg_prompts, is_sdxl=False)
    pipeline(**kwargs, width=..., ...)

Usage (SDXL):
    kwargs = build_prompt_embeds(pipeline, prompts, neg_prompts, is_sdxl=True)
    pipeline(**kwargs, width=..., ...)
"""
from __future__ import annotations

import torch


def build_prompt_embeds(
    pipeline,
    prompts: list[str],
    neg_prompts: list[str],
    *,
    is_sdxl: bool,
) -> dict:
    """
    Encode *prompts* and *neg_prompts* with Compel and return a kwargs dict
    ready to be unpacked into the diffusers pipeline call.

    For SD 1.x the dict contains:
        prompt_embeds, negative_prompt_embeds

    For SDXL the dict additionally contains:
        pooled_prompt_embeds, negative_pooled_prompt_embeds

    The *prompt* / *negative_prompt* string keys are intentionally absent so
    that diffusers uses the pre-computed embeddings path and does not re-encode.

    Args:
        pipeline:    a loaded diffusers pipeline with tokenizer(s) and
                     text_encoder(s) available.
        prompts:     positive prompt strings, one per batch item.
        neg_prompts: negative prompt strings, same length as *prompts*.
        is_sdxl:     True for SDXL (dual-encoder), False for SD 1.x / 2.x.

    Returns:
        dict of pipeline keyword arguments.
    """
    from compel import Compel, ReturnedEmbeddingsType

    if is_sdxl:
        # SDXL has two text encoders:
        #   tokenizer  + text_encoder   → CLIP ViT-L  (no pooled output used)
        #   tokenizer_2 + text_encoder_2 → OpenCLIP ViT-bigG (pooled output used)
        compel_proc = Compel(
            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        pos_embeds, pos_pooled = _encode_batch_sdxl(compel_proc, prompts)
        neg_embeds, neg_pooled = _encode_batch_sdxl(compel_proc, neg_prompts)
        return {
            "prompt_embeds": pos_embeds,
            "negative_prompt_embeds": neg_embeds,
            "pooled_prompt_embeds": pos_pooled,
            "negative_pooled_prompt_embeds": neg_pooled,
        }
    else:
        # SD 1.x / 2.x: single tokenizer + text encoder
        compel_proc = Compel(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            truncate_long_prompts=False,
        )
        pos_embeds = _encode_batch_sd1(compel_proc, prompts)
        neg_embeds = _encode_batch_sd1(compel_proc, neg_prompts)
        return {
            "prompt_embeds": pos_embeds,
            "negative_prompt_embeds": neg_embeds,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_batch_sd1(compel_proc, texts: list[str]) -> torch.Tensor:
    """
    Encode *texts* individually (to handle distinct prompt lengths), then pad
    all embeddings to the same sequence length and concatenate into a single
    batch tensor of shape (N, seq_len, hidden_size).
    """
    embeddings = [compel_proc(t) for t in texts]
    if len(embeddings) == 1:
        return embeddings[0]
    return compel_proc.pad_conditioning_tensors_to_same_length(embeddings)


def _encode_batch_sdxl(
    compel_proc, texts: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode *texts* for SDXL individually, pad the sequence-dimension embeddings,
    and stack the pooled embeddings into a single batch tensor.

    Returns:
        (padded_embeds, pooled)  — shapes (N, seq_len, hidden) and (N, pooled_dim)
    """
    embeds_list: list[torch.Tensor] = []
    pooled_list: list[torch.Tensor] = []

    for t in texts:
        embeds, pooled = compel_proc(t)
        embeds_list.append(embeds)
        pooled_list.append(pooled)

    if len(embeds_list) == 1:
        return embeds_list[0], pooled_list[0]

    padded_embeds = compel_proc.pad_conditioning_tensors_to_same_length(embeds_list)
    pooled = torch.cat(pooled_list, dim=0)
    return padded_embeds, pooled

