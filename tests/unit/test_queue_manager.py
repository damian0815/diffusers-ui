"""Unit tests for backend/queue/manager.py"""
import asyncio
import pytest
from backend.queue.manager import QueueManager
from backend.domain import QueueItem, ItemStatus, make_params


@pytest.fixture
def qm():
    return QueueManager()


@pytest.mark.asyncio
async def test_enqueue_append_order(qm):
    a = await qm.enqueue(QueueItem(params=make_params()))
    b = await qm.enqueue(QueueItem(params=make_params()))
    items = await qm.next_pending_batch_window(0)
    assert items[0].item_id == a.item_id
    assert items[1].item_id == b.item_id


@pytest.mark.asyncio
async def test_enqueue_orders_by_priority_value(qm):
    """Higher priority_value items are dequeued first; equal ranks stay FIFO."""
    a = await qm.enqueue(QueueItem(params=make_params(), priority_value=1))
    c = await qm.enqueue(QueueItem(params=make_params(), priority_value=2))
    b = await qm.enqueue(QueueItem(params=make_params(), priority_value=1))
    items = await qm.next_pending_batch_window(0)
    assert [i.item_id for i in items] == [c.item_id, a.item_id, b.item_id]


@pytest.mark.asyncio
async def test_move_to_front_bumps_priority_value(qm):
    """move_to_front raises the item's rank to max+1, putting it at the head."""
    high = await qm.enqueue(QueueItem(params=make_params(), priority_value=3))
    a = await qm.enqueue(QueueItem(params=make_params(), priority_value=1))
    b = await qm.enqueue(QueueItem(params=make_params(), priority_value=1))
    assert await qm.move_to_front(b.item_id) is True
    assert b.priority_value == high.priority_value + 1
    items = await qm.next_pending_batch_window(0)
    assert [i.item_id for i in items] == [b.item_id, high.item_id, a.item_id]


@pytest.mark.asyncio
async def test_move_to_front_rejects_non_pending(qm):
    item = await qm.enqueue(QueueItem(params=make_params()))
    await qm.mark_processing(item)
    assert await qm.move_to_front(item.item_id) is False
    assert await qm.move_to_front(99999) is False


@pytest.mark.asyncio
async def test_requeue_items_restores_order_by_rank(qm):
    """Requeued items re-enter PENDING and sort by their existing rank."""
    a = await qm.enqueue(QueueItem(params=make_params(), priority_value=2))
    b = await qm.enqueue(QueueItem(params=make_params(), priority_value=2))
    # Drain the deque the way the worker does, then mark them processing.
    await qm.next_pending_batch_window(0)
    await qm.mark_processing(a)
    await qm.mark_processing(b)
    await qm.requeue_items([a, b])
    assert a.status == ItemStatus.PENDING
    assert b.status == ItemStatus.PENDING
    items = await qm.next_pending_batch_window(0)
    assert [i.item_id for i in items] == [a.item_id, b.item_id]


@pytest.mark.asyncio
async def test_cancel_pending_item(qm):
    item = await qm.enqueue(QueueItem(params=make_params()))
    result = await qm.cancel_item(item.item_id)
    assert result is True
    assert qm.get_item(item.item_id).status == ItemStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_processing_item_returns_false(qm):
    item = await qm.enqueue(QueueItem(params=make_params()))
    await qm.mark_processing(item)
    result = await qm.cancel_item(item.item_id)
    assert result is False
    assert qm.get_item(item.item_id).status == ItemStatus.PROCESSING


@pytest.mark.asyncio
async def test_cancel_nonexistent_item_returns_false(qm):
    result = await qm.cancel_item(99999)
    assert result is False


@pytest.mark.asyncio
async def test_cancel_all_only_affects_pending(qm):
    pending = await qm.enqueue(QueueItem(params=make_params()))
    processing = await qm.enqueue(QueueItem(params=make_params()))
    await qm.mark_processing(processing)
    count = await qm.cancel_all()
    assert len(count) == 1
    assert count == [pending]
    assert qm.get_item(pending.item_id).status == ItemStatus.CANCELLED
    assert qm.get_item(processing.item_id).status == ItemStatus.PROCESSING


@pytest.mark.asyncio
async def test_cancelled_items_excluded_from_batch_window(qm):
    item = await qm.enqueue(QueueItem(params=make_params()))
    await qm.cancel_item(item.item_id)
    # Enqueue another so event is set
    good = await qm.enqueue(QueueItem(params=make_params()))
    items = await qm.next_pending_batch_window(0)
    assert all(i.item_id != item.item_id for i in items)
    assert any(i.item_id == good.item_id for i in items)


@pytest.mark.asyncio
async def test_mark_completed(qm):
    item = await qm.enqueue(QueueItem(params=make_params()))
    await qm.mark_processing(item)
    await qm.mark_completed(item, "output.png")
    assert item.status == ItemStatus.COMPLETED
    assert item.output_image_name == "output.png"


@pytest.mark.asyncio
async def test_mark_failed(qm):
    item = await qm.enqueue(QueueItem(params=make_params()))
    await qm.mark_processing(item)
    await qm.mark_failed(item, "CUDA OOM")
    assert item.status == ItemStatus.FAILED
    assert item.error == "CUDA OOM"


@pytest.mark.asyncio
async def test_prune_removes_terminal_items(qm):
    item_pending = await qm.enqueue(QueueItem(params=make_params()))
    item_done = await qm.enqueue(QueueItem(params=make_params()))
    item_failed = await qm.enqueue(QueueItem(params=make_params()))
    await qm.mark_processing(item_done)
    await qm.mark_completed(item_done, "img.png")
    await qm.mark_processing(item_failed)
    await qm.mark_failed(item_failed, "error")

    pruned = await qm.prune()
    assert pruned == 2
    assert qm.get_item(item_pending.item_id) is not None
    assert qm.get_item(item_done.item_id) is None
    assert qm.get_item(item_failed.item_id) is None


@pytest.mark.asyncio
async def test_all_items_returns_all_tracked(qm):
    a = await qm.enqueue(QueueItem(params=make_params()))
    b = await qm.enqueue(QueueItem(params=make_params()))
    ids = {i.item_id for i in qm.all_items()}
    assert a.item_id in ids
    assert b.item_id in ids


@pytest.mark.asyncio
async def test_batch_window_waits_for_items():
    """next_pending_batch_window should block until items are available."""
    qm = QueueManager()

    async def add_later():
        await asyncio.sleep(0.05)
        await qm.enqueue(QueueItem(params=make_params()))

    task = asyncio.create_task(add_later())
    items = await asyncio.wait_for(qm.next_pending_batch_window(0), timeout=2.0)
    assert len(items) == 1
    await task


@pytest.mark.asyncio
async def test_batch_window_prefers_loaded_model(qm):
    """When a preferred model is given, an item using it is picked as the pivot
    even if it is not at the head of the queue."""
    head = await qm.enqueue(QueueItem(params=make_params(model_key="model_b")))
    preferred = await qm.enqueue(QueueItem(params=make_params(model_key="model_a")))
    other = await qm.enqueue(QueueItem(params=make_params(model_key="model_a")))

    items = await qm.next_pending_batch_window(0, limit=1, preferred_model_key="model_a")
    assert [i.item_id for i in items] == [preferred.item_id]

    # The skipped head stays in the queue for the next call.
    next_items = await qm.next_pending_batch_window(0, limit=1)
    assert next_items[0].item_id == head.item_id


@pytest.mark.asyncio
async def test_batch_window_preference_falls_back_to_fifo(qm):
    """If no pending item uses the preferred model, the queue head is returned."""
    a = await qm.enqueue(QueueItem(params=make_params(model_key="model_b")))
    b = await qm.enqueue(QueueItem(params=make_params(model_key="model_b")))
    items = await qm.next_pending_batch_window(0, limit=1, preferred_model_key="model_a")
    assert [i.item_id for i in items] == [a.item_id]
    assert items[0].item_id != b.item_id


@pytest.mark.asyncio
async def test_batch_window_preference_never_skips_higher_rank(qm):
    """A higher-ranked item at the head wins over the model preference."""
    high = await qm.enqueue(QueueItem(params=make_params(model_key="model_b"), priority_value=5))
    await qm.enqueue(QueueItem(params=make_params(model_key="model_a"), priority_value=1))
    await qm.enqueue(QueueItem(params=make_params(model_key="model_a"), priority_value=1))
    items = await qm.next_pending_batch_window(0, limit=1, preferred_model_key="model_a")
    assert items[0].item_id == high.item_id


@pytest.mark.asyncio
async def test_batch_window_preference_scan_is_bounded(monkeypatch):
    """The preferred-model scan only looks a bounded distance from the head,
    so a long run of another model cannot be starved."""
    from backend.queue import manager as qm_mod
    monkeypatch.setattr(qm_mod, "PREFERRED_MODEL_SCAN_LIMIT", 2)
    qm = QueueManager()
    for _ in range(2):
        await qm.enqueue(QueueItem(params=make_params(model_key="model_b")))
    target = await qm.enqueue(QueueItem(params=make_params(model_key="model_a")))
    items = await qm.next_pending_batch_window(0, limit=1, preferred_model_key="model_a")
    assert items[0].item_id != target.item_id  # outside the scan window

