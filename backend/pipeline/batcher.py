from __future__ import annotations

from typing import Callable, Union

from backend.domain import ImageGenerationParams, QueueItem, CriticalParams


def get_critical_params(params: ImageGenerationParams) -> CriticalParams:
    """
    Extract the params that must match for two items to be batched together.
    Items with different critical params require separate pipeline calls.
    """
    return CriticalParams(
        width=params.width,
        height=params.height,
        steps=params.steps,
        cfg=params.cfg,
        cfg_rescale_multiplier=params.cfg_rescale_multiplier,
        sampler=params.sampler,
        model_key=params.model_key,
        lora_keys=tuple(params.lora_keys),
    )


def group_into_batches(
    items: list[QueueItem],
    max_size: Union[int, Callable[[CriticalParams], int]] = 0,
) -> list[list[QueueItem]]:
    """
    Group items by critical params *and priority rank*, preserving queue order
    within each group.  Returns a list of groups; each group can be sent to
    the pipeline as one batch.  Chunking by rank guarantees a batch never
    mixes priority levels.

    Args:
        items:    flat list of pending QueueItems from the queue.
        max_size: if an int > 0, split any group that exceeds this size into
                  chunks of at most *max_size* items.
                  If callable, it is called with the group's CriticalParams and
                  should return the maximum batch size for that group (0 =
                  unlimited).  This allows per-resolution limits derived from
                  empirical memory measurements.

    Example:
        [A(512x512, rank 1), B(768x768, rank 1), C(512x512, rank 2)] →
        [[A], [B], [C]]    (A and C share params but not rank)
    """
    seen: dict[tuple[CriticalParams, int], list[QueueItem]] = {}
    for item in items:
        key = (get_critical_params(item.params), item.priority_value)
        seen.setdefault(key, []).append(item)

    result: list[list[QueueItem]] = []
    for (crit, _rank), group in seen.items():
        limit: int = max_size(crit) if callable(max_size) else max_size
        if limit > 0 and len(group) > limit:
            for i in range(0, len(group), limit):
                result.append(group[i : i + limit])
        else:
            result.append(group)

    return result
