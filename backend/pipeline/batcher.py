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
    )


def group_into_batches(
    items: list[QueueItem],
    max_size: Union[int, Callable[[CriticalParams], int]] = 0,
) -> list[list[QueueItem]]:
    """
    Group items by critical params, preserving queue order within each group.
    Returns a list of groups; each group can be sent to the pipeline as one batch.

    Args:
        items:    flat list of pending QueueItems from the queue.
        max_size: if an int > 0, split any group that exceeds this size into
                  chunks of at most *max_size* items.
                  If callable, it is called with the group's CriticalParams and
                  should return the maximum batch size for that group (0 =
                  unlimited).  This allows per-resolution limits derived from
                  empirical memory measurements.

    Example:
        [A(512x512), B(768x768), C(512x512)] →
        [[A, C], [B]]    (A and C share critical params)
    """
    seen: dict[CriticalParams, list[QueueItem]] = {}
    for item in items:
        key = get_critical_params(item.params)
        seen.setdefault(key, []).append(item)

    result: list[list[QueueItem]] = []

    # ── Priority items: each runs as its own solo batch (no grouping) ──────────
    # They are placed first so they execute before any normal-priority work that
    # was already in the window.
    priority_result: list[list[QueueItem]] = []
    normal_result: list[list[QueueItem]] = []

    for key, group in seen.items():
        # Split the group into priority singles and normal chunks
        for item in group:
            if item.priority:
                priority_result.append([item])
        normal_group = [item for item in group if not item.priority]
        if normal_group:
            limit: int = max_size(key) if callable(max_size) else max_size
            if limit > 0 and len(normal_group) > limit:
                for i in range(0, len(normal_group), limit):
                    normal_result.append(normal_group[i : i + limit])
            else:
                normal_result.append(normal_group)

    return priority_result + normal_result
