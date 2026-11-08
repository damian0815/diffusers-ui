"""Unit tests for backend/queue/listing.py — pure, I/O-free helpers."""
from backend.domain import ItemStatus, QueueItem, make_params
from backend.queue.listing import (
    filter_items,
    list_models,
    paginate,
    priority_count,
    sort_queue,
    summarize,
)


def make_item(prompt="a beautiful landscape", model_key="test_model", item_id=None, **overrides):
    params = make_params(prompt=prompt, model_key=model_key)
    kwargs = dict(params=params)
    if item_id is not None:
        kwargs["item_id"] = item_id
    kwargs.update(overrides)
    return QueueItem(**kwargs)


class FakeRegistry:
    """Dict-like registry: entries with .name, missing keys -> None."""

    def __init__(self, entries: dict):
        self._entries = entries

    def get(self, key):
        return self._entries.get(key)


def named(name):
    return type("Entry", (), {"name": name})


# ---------------------------------------------------------------------------
# filter_items
# ---------------------------------------------------------------------------

def test_filter_items_by_single_status():
    items = [
        make_item(item_id=1, status=ItemStatus.PENDING),
        make_item(item_id=2, status=ItemStatus.COMPLETED),
        make_item(item_id=3, status=ItemStatus.CANCELLED),
    ]
    result = filter_items(items, statuses={"pending"})
    assert [i.item_id for i in result] == [1]


def test_filter_items_by_status_set():
    items = [
        make_item(item_id=1, status=ItemStatus.PENDING),
        make_item(item_id=2, status=ItemStatus.COMPLETED),
        make_item(item_id=3, status=ItemStatus.FAILED),
    ]
    result = filter_items(items, statuses={"completed", "failed"})
    assert [i.item_id for i in result] == [2, 3]


def test_filter_items_by_model_key():
    items = [
        make_item(item_id=1, model_key="alpha"),
        make_item(item_id=2, model_key="beta"),
    ]
    result = filter_items(items, model_key="beta")
    assert [i.item_id for i in result] == [2]


def test_filter_items_by_prompt_case_insensitive():
    items = [
        make_item(item_id=1, prompt="a beautiful landscape"),
        make_item(item_id=2, prompt="ugly cityscape"),
    ]
    result = filter_items(items, q="BEAUTIFUL")
    assert [i.item_id for i in result] == [1]


def test_filter_items_whitespace_only_q_is_noop():
    items = [make_item(item_id=1, prompt="anything"), make_item(item_id=2, prompt="else")]
    assert len(filter_items(items, q="   ")) == 2
    assert len(filter_items(items, q="")) == 2


def test_filter_items_and_combination():
    items = [
        make_item(item_id=1, prompt="a cat", model_key="alpha", status=ItemStatus.PENDING),
        make_item(item_id=2, prompt="a cat", model_key="beta", status=ItemStatus.PENDING),
        make_item(item_id=3, prompt="a dog", model_key="alpha", status=ItemStatus.PENDING),
        make_item(item_id=4, prompt="a cat", model_key="alpha", status=ItemStatus.COMPLETED),
    ]
    result = filter_items(items, statuses={"pending"}, model_key="alpha", q="cat")
    assert [i.item_id for i in result] == [1]


def test_filter_items_no_filters_returns_all_in_order():
    items = [make_item(item_id=1), make_item(item_id=2), make_item(item_id=3)]
    assert filter_items(items) == items


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_counts_per_status():
    items = [
        make_item(item_id=1, status=ItemStatus.PENDING),
        make_item(item_id=2, status=ItemStatus.PENDING),
        make_item(item_id=3, status=ItemStatus.PROCESSING),
        make_item(item_id=4, status=ItemStatus.COMPLETED),
        make_item(item_id=5, status=ItemStatus.COMPLETED),
        make_item(item_id=6, status=ItemStatus.FAILED),
        make_item(item_id=7, status=ItemStatus.CANCELLED),
    ]
    assert summarize(items) == {
        "pending": 2,
        "processing": 1,
        "completed": 2,
        "failed": 1,
        "cancelled": 1,
    }


def test_summarize_empty_list():
    assert summarize([]) == {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

def test_list_models_distinct_keys_with_counts_and_names():
    registry = FakeRegistry({"alpha": named("Alpha model")})
    items = [
        make_item(item_id=1, model_key="alpha"),
        make_item(item_id=2, model_key="alpha"),
        make_item(item_id=3, model_key="beta"),
    ]
    result = list_models(items, registry)
    assert result == [
        {"model_key": "alpha", "model_name": "Alpha model", "count": 2},
        {"model_key": "beta", "model_name": "beta", "count": 1},
    ]


def test_list_models_missing_registry_entry_falls_back_to_raw_key():
    registry = FakeRegistry({})
    items = [make_item(item_id=1, model_key="gamma")]
    result = list_models(items, registry)
    assert result[0]["model_name"] == "gamma"
    assert result[0]["count"] == 1


def test_list_models_empty():
    assert list_models([], FakeRegistry({})) == []


# ---------------------------------------------------------------------------
# paginate
# ---------------------------------------------------------------------------

def test_paginate_page_smaller_than_total():
    items = [make_item(item_id=i) for i in range(1, 6)]
    page, has_more = paginate(items, 0, 2)
    assert [i.item_id for i in page] == [1, 2]
    assert has_more is True


def test_paginate_exact_boundary_page():
    items = [make_item(item_id=i) for i in range(1, 5)]
    page, has_more = paginate(items, 0, 4)
    assert len(page) == 4
    assert has_more is False


def test_paginate_offset_past_end():
    items = [make_item(item_id=i) for i in range(1, 4)]
    page, has_more = paginate(items, 10, 2)
    assert page == []
    assert has_more is False


def test_paginate_limit_zero_metadata_only():
    items = [make_item(item_id=i) for i in range(1, 4)]
    page, has_more = paginate(items, 0, 0)
    assert page == []
    assert has_more is True


def test_paginate_limit_zero_empty_items():
    page, has_more = paginate([], 0, 0)
    assert page == []
    assert has_more is False


# ---------------------------------------------------------------------------
# sort_queue
# ---------------------------------------------------------------------------

def test_sort_queue_priority_before_regular():
    regular = make_item(item_id=1, priority=False)
    priority = make_item(item_id=2, priority=True)
    result = sort_queue([regular, priority])
    assert [i.item_id for i in result] == [2, 1]


def test_sort_queue_item_id_ascending_within_groups():
    items = [
        make_item(item_id=10, priority=False),
        make_item(item_id=5, priority=True),
        make_item(item_id=7, priority=True),
        make_item(item_id=1, priority=False),
    ]
    result = sort_queue(items)
    assert [i.item_id for i in result] == [5, 7, 1, 10]


def test_sort_queue_stable_for_equal_keys():
    a = make_item(item_id=3, priority=True)
    b = make_item(item_id=3, priority=True)
    result = sort_queue([b, a])
    assert result == [b, a]


def test_sort_queue_higher_priority_value_first():
    low = make_item(item_id=1, priority_value=1)
    high = make_item(item_id=2, priority_value=3)
    result = sort_queue([low, high])
    assert [i.item_id for i in result] == [2, 1]


def test_sort_queue_run_now_still_leads_higher_rank():
    """A prepend/run-now item stays above plain items regardless of rank."""
    run_now = make_item(item_id=1, priority=True, priority_value=1)
    ranked = make_item(item_id=2, priority_value=9)
    result = sort_queue([ranked, run_now])
    assert [i.item_id for i in result] == [1, 2]


def test_sort_queue_submission_serial_breaks_equal_rank():
    a = make_item(item_id=10, priority_value=2)
    b = make_item(item_id=7, priority_value=2)
    result = sort_queue([a, b])
    assert [i.item_id for i in result] == [7, 10]


def test_sort_queue_mixed_ranks_full_order():
    items = [
        make_item(item_id=1, priority_value=1),
        make_item(item_id=2, priority=True, priority_value=1),
        make_item(item_id=3, priority_value=2),
        make_item(item_id=4, priority_value=9),
        make_item(item_id=5, priority_value=2),
    ]
    result = sort_queue(items)
    assert [i.item_id for i in result] == [2, 4, 3, 5, 1]


# ---------------------------------------------------------------------------
# priority_count
# ---------------------------------------------------------------------------

def test_priority_count_zero_when_none():
    items = [make_item(item_id=1, priority=False), make_item(item_id=2, priority=False)]
    assert priority_count(items) == 0


def test_priority_count_some():
    items = [
        make_item(item_id=1, priority=False),
        make_item(item_id=2, priority=True),
        make_item(item_id=3, priority=True),
    ]
    assert priority_count(items) == 2
