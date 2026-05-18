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
    max_size: int = 0,
) -> list[list[QueueItem]]:
    """
    Group items by critical params, preserving queue order within each group.
    Returns a list of groups; each group can be sent to the pipeline as one batch.

    Args:
        items:    flat list of pending QueueItems from the queue.
        max_size: if > 0, split any group that exceeds this size into chunks of
                  at most *max_size* items.  Used to reduce VRAM pressure after
                  an OOM failure.

    Example:
        [A(512x512), B(768x768), C(512x512)] →
        [[A, C], [B]]    (A and C share critical params)
    """
    seen: dict[CriticalParams, list[QueueItem]] = {}
    for item in items:
        key = get_critical_params(item.params)
        seen.setdefault(key, []).append(item)

    result: list[list[QueueItem]] = []
    for group in seen.values():
        if max_size > 0 and len(group) > max_size:
            for i in range(0, len(group), max_size):
                result.append(group[i : i + max_size])
        else:
            result.append(group)
    return result
