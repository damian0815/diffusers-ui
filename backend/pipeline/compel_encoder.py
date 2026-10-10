"""
Compel-based prompt encoding for SD 1.x / 2.x, SDXL and SANA pipelines.

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

Usage (SANA):
    kwargs = build_sana_prompt_embeds(pipeline, prompts, neg_prompts)
    pipeline(**kwargs, width=..., ...)
"""
from __future__ import annotations

import torch
from compel import CompelForSDXL

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
        # SDXL requires two *separate* Compel instances — one per text encoder.
        # Using a single Compel with list inputs causes it to return a Python
        # list of per-encoder tensors instead of a tensor, which breaks
        # pad_conditioning_tensors_to_same_length (.shape access fails).
        #
        # Pattern mirrors CompelForSDXL in compel.convenience_wrappers:
        #   compel_1 → CLIP ViT-L  (penultimate, no pooled, 768-dim)
        #   compel_2 → OpenCLIP ViT-bigG (penultimate + pooled, 1280-dim)
        # The two conditioning tensors are seq-length-aligned then concatenated
        # on the hidden dimension → [B, S, 2048] expected by the SDXL pipeline.
        compel = CompelForSDXL(pipeline)
        #compel_1 = Compel(
        #    tokenizer=pipeline.tokenizer,
        #    text_encoder=pipeline.text_encoder,
        #    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        #    truncate_long_prompts=False,
        #)
        #compel_2 = Compel(
        #    tokenizer=pipeline.tokenizer_2,
        #    text_encoder=pipeline.text_encoder_2,
        #    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        #    requires_pooled=True,
        #    truncate_long_prompts=False,
        #)
        #pos_embeds, pos_pooled = _encode_batch_sdxl(compel.compel_1, compel.compel_2, prompts)
        #neg_embeds, neg_pooled = _encode_batch_sdxl(compel.compel_1, compel.compel_2, neg_prompts)

        #full_padded = compel.compel_2.pad_conditioning_tensors_to_same_length(torch.cat([pos_embeds, neg_embeds]))
        #split_pos = pos_embeds.shape[0]
        #pos_embeds = full_padded[0:split_pos]
        #neg_embeds = full_padded[split_pos:]


        #compel.disable_no_weights_bypass()
        embeds_fancy = compel(prompts, negative_prompt=neg_prompts)
        pos_embeds, pos_pooled = embeds_fancy.embeds, embeds_fancy.pooled_embeds
        neg_embeds, neg_pooled = embeds_fancy.negative_embeds, embeds_fancy.negative_pooled_embeds

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_batch_sd1(compel_proc, texts: list[str]) -> torch.Tensor:
    """
    Encode *texts* individually then pad all to the same sequence length.
    Returns a single batched tensor of shape (N, seq_len, hidden_size).
    """
    embeddings = [compel_proc(t) for t in texts]
    if len(embeddings) == 1:
        return embeddings[0]
    # pad_conditioning_tensors_to_same_length returns List[Tensor]; must cat.
    padded = compel_proc.pad_conditioning_tensors_to_same_length(embeddings)
    return torch.cat(padded, dim=0)


def _encode_batch_sdxl(
    compel_1: object,
    compel_2: object,
    texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode *texts* for SDXL using two separate Compel instances.

    Each text is encoded by both encoders independently.  The resulting
    left (768-d) and right (1280-d) tensors are:
      1. Each padded within their own encoder to the same seq length.
      2. Cross-aligned to each other (the two tokenizers may produce
         different sequence lengths for the same text even at the same
         chunk count).
      3. Concatenated on the hidden dimension → [B, S, 2048].

    Returns:
        (prompt_embeds [B, S, 2048],  pooled_prompt_embeds [B, 1280])
    """
    left_list:   list[torch.Tensor] = []
    right_list:  list[torch.Tensor] = []
    pooled_list: list[torch.Tensor] = []

    for t in texts:
        left = compel_1(t)            # [1, S_a, 768]
        right, pooled = compel_2(t)   # [1, S_b, 1280],  [1, 1280]
        left_list.append(left)
        right_list.append(right)
        pooled_list.append(pooled)

    # 1. Pad within each encoder's batch to a common seq length.
    #    pad_conditioning_tensors_to_same_length returns List[Tensor]; must cat.
    if len(left_list) == 1:
        left_padded  = left_list[0]
        right_padded = right_list[0]
        pooled       = pooled_list[0]
    else:
        left_padded  = torch.cat(compel_1.pad_conditioning_tensors_to_same_length(left_list),  dim=0)
        right_padded = torch.cat(compel_2.pad_conditioning_tensors_to_same_length(right_list), dim=0)
        pooled       = torch.cat(pooled_list, dim=0)

    # 2. Align the two sides to the same sequence length.
    #    (The two tokenizers can produce different lengths for the same text.)
    s_l, s_r = left_padded.shape[1], right_padded.shape[1]
    if s_l < s_r:
        pad = torch.zeros(
            left_padded.shape[0], s_r - s_l, left_padded.shape[2],
            dtype=left_padded.dtype, device=left_padded.device,
        )
        left_padded = torch.cat([left_padded, pad], dim=1)
    elif s_r < s_l:
        pad = torch.zeros(
            right_padded.shape[0], s_l - s_r, right_padded.shape[2],
            dtype=right_padded.dtype, device=right_padded.device,
        )
        right_padded = torch.cat([right_padded, pad], dim=1)

    # 3. Concatenate on the hidden dimension → [B, S, 768+1280] = [B, S, 2048].
    embeds = torch.cat([left_padded, right_padded], dim=-1)
    return embeds, pooled
