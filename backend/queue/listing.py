"""Pure, synchronous helpers for listing / filtering the queue.

Kept I/O-free so the route handler stays thin and every function here is
unit-testable in isolation (per AGENTS.md).
"""
from __future__ import annotations

from typing import Iterable

from backend.domain import ItemStatus, QueueItem


def filter_items(
    items,
    statuses=None,
    model_key=None,
    q=None,
) -> list[QueueItem]:
    """AND-combined filters applied to *items*:

    - statuses: set[str] | None -> keep item.status.value in statuses
    - model_key: str | None     -> exact match on item.params.model_key
    - q: str | None             -> case-insensitive substring on item.params.prompt
                                   (whitespace-only / empty treated as None)
    """
    statuses = set(statuses) if statuses else None
    query = (q or "").strip().lower()
    result = []
    for item in items:
        if statuses is not None and item.status.value not in statuses:
            continue
        if model_key is not None and item.params.model_key != model_key:
            continue
        if query and query not in item.params.prompt.lower():
            continue
        result.append(item)
    return result


def sort_queue(items) -> list[QueueItem]:
    """Priority items first, then item_id ascending within each group."""
    return sorted(items, key=lambda it: (not it.priority, it.item_id))


def summarize(items) -> dict[str, int]:
    """Per-status counts: {pending, processing, completed, failed, cancelled}."""
    counts = {status.value: 0 for status in ItemStatus}
    for item in items:
        counts[item.status.value] += 1
    return counts


def list_models(items, registry) -> list[dict]:
    """Distinct model_key across *items*, each as
    {model_key, model_name, count}.
    model_name = registry.get(key).name if present else the raw key.
    """
    counts: dict[str, int] = {}
    for item in items:
        key = item.params.model_key
        counts[key] = counts.get(key, 0) + 1

    result = []
    for key, count in counts.items():
        entry = registry.get(key)
        name = entry.name if entry else key
        result.append({"model_key": key, "model_name": name, "count": count})
    return result


def paginate(items, offset, limit) -> tuple[list, bool]:
    """Return (page_slice, has_more). Guard offset >= len -> empty page,
    has_more False. limit == 0 -> ([], True if items else False) — a
    metadata-only fetch (no items slice)."""
    if limit == 0:
        return ([], True if items else False)
    if offset >= len(items):
        return ([], False)
    end = offset + limit
    return (items[offset:end], end < len(items))


def priority_count(items) -> int:
    """Number of items with item.priority is True."""
    return sum(1 for it in items if it.priority)
