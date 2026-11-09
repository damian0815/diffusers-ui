import asyncio
import logging
from collections import deque

from backend.domain import QueueItem, ItemStatus

logger = logging.getLogger(__name__)

QUEUE_POLL_INTERVAL_S = 0.05

# How many slots of the queue head we may scan for a preferred-model item
# before falling back to strict FIFO.  Bounds the reordering so a long run of
# another model can't be starved by sustained same-model submissions.
PREFERRED_MODEL_SCAN_LIMIT = 32


class QueueManager:
    """
    Manages a queue of QueueItems.

    Pass ``store`` to enable persistence (DB-backed); omit for pure in-memory
    operation (used in tests).
    """

    def __init__(self, store=None):
        self._queue: deque[QueueItem] = deque()
        self._all_items: dict[int, QueueItem] = {}
        self._lock = asyncio.Lock()
        self._new_item_event = asyncio.Event()
        self._shutting_down = False
        self._store = store  # optional QueueStore

    # ------------------------------------------------------------------
    # Persistence helpers (best-effort; failures are logged, not raised)
    # ------------------------------------------------------------------

    async def _persist(self, item: QueueItem) -> None:
        if self._store is None:
            return
        try:
            await self._store.upsert(item)
        except Exception:
            logger.warning("Failed to persist queue item %d", item.item_id, exc_info=True)

    async def enqueue(self, item: QueueItem) -> QueueItem:
        """Add an item to the queue, in priority order."""
        async with self._lock:
            self._insert_in_priority_order(item)
            self._all_items[item.item_id] = item
        await self._persist(item)
        self._new_item_event.set()
        return item

    def _insert_in_priority_order(self, item: QueueItem) -> None:
        """Insert an item so the deque stays ordered by
        (descending priority_value, ascending item_id).

        Ordering is driven solely by priority_value — higher priority_value
        lands closer to the head, so the worker's head-first pickup processes
        items in priority order.  Within one rank the deque keeps submission
        order (FIFO) because new items always carry the largest item_id.
        """
        insert_at = len(self._queue)
        for idx in range(len(self._queue)):
            if self._queue[idx].priority_value < item.priority_value:
                insert_at = idx
                break
        self._queue.insert(insert_at, item)

    async def requeue_items(self, items: list[QueueItem]) -> None:
        """
        Reset *items* to PENDING and put them back in the queue.

        Used after an OOM failure to retry at a smaller batch size.  Items
        re-enter in their original relative order, positioned by their existing
        priority_value, so they are picked up before lower-ranked work.
        """
        async with self._lock:
            for item in items:
                item.status = ItemStatus.PENDING
                item.error = None
                self._insert_in_priority_order(item)
        for item in items:
            await self._persist(item)
        self._new_item_event.set()

    async def move_to_front(self, item_id: int) -> bool:
        """
        Move a *pending* item to the front of the queue.

        Ordering is driven solely by priority_value, so this bumps the item's
        rank above every other tracked item (max + 1) and re-inserts it in
        priority order.

        Returns False if the item is not found or not in PENDING status.
        """
        async with self._lock:
            item = self._all_items.get(item_id)
            if item is None or item.status != ItemStatus.PENDING:
                return False
            try:
                self._queue.remove(item)
            except ValueError:
                return False  # item not in deque (shouldn't happen for PENDING)
            item.priority_value = max(
                (it.priority_value for it in self._all_items.values()), default=0
            ) + 1
            self._insert_in_priority_order(item)
        await self._persist(item)
        return True

    async def next_pending_batch_window(
        self,
        window_s: float,
        limit: int=100,
        preferred_model_key: str | None = None,
    ) -> list[QueueItem]:
        """
        Wait until at least one pending item is available, then collect all
        items that arrive within ``window_s`` seconds.  Returns an empty list
        if ``abort_wait()`` was called (shutdown signal).

        If *preferred_model_key* is given, the first collected item is chosen
        with ``_pop_preferred_pivot`` — a pending item that uses that model and
        shares the head's priority rank, so the worker can minimise model
        swaps without reordering across priority levels.  Falls back to the
        queue head when no preferred item is found.
        """
        await self._new_item_event.wait()

        if self._shutting_down:
            return []

        if window_s > 0:
            await asyncio.sleep(window_s)

        async with self._lock:
            self._new_item_event.clear()
            collected: list[QueueItem] = []
            if preferred_model_key is not None:
                pivot = self._pop_preferred_pivot(preferred_model_key)
                if pivot is not None:
                    collected.append(pivot)
            while self._queue and len(collected) < limit:
                item = self._queue.popleft()
                if item.status == ItemStatus.PENDING:
                    collected.append(item)
            if not self._queue:
                self._new_item_event.clear()
            else:
                self._new_item_event.set()
        return collected

    def _pop_preferred_pivot(self, preferred_model_key: str) -> QueueItem | None:
        """
        Pop the first pending item that uses *preferred_model_key*, scanning at
        most ``PREFERRED_MODEL_SCAN_LIMIT`` slots from the head.

        Ordering is driven solely by priority_value, so the scan may only
        reorder *within* the leading priority rank.  As soon as it steps into a
        lower-ranked block it gives up and returns None (the caller falls back
        to the queue head), so a preferred model can never overtake a
        higher-ranked item.
        """
        scan = min(PREFERRED_MODEL_SCAN_LIMIT, len(self._queue))
        head_rank: int | None = None
        for i in range(scan):
            item = self._queue[i]
            if item.status != ItemStatus.PENDING:
                continue
            if head_rank is None:
                head_rank = item.priority_value
            if item.priority_value != head_rank:
                return None  # left the leading rank block — don't reorder
            if item.params.model_key == preferred_model_key:
                del self._queue[i]
                return item
        return None

    async def dequeue_matching(
        self,
        predicate,
        limit: int,
    ) -> list[QueueItem]:
        """
        Scan the entire queue without waiting, and remove up to *limit*
        pending items for which *predicate(item)* returns True.

        Non-matching pending items stay in their original order.
        Non-pending items are silently dropped.
        """
        async with self._lock:
            collected: list[QueueItem] = []
            remaining: deque[QueueItem] = deque()
            while self._queue and len(collected) < limit:
                item = self._queue.popleft()
                if item.status == ItemStatus.PENDING and predicate(item):
                    collected.append(item)
                elif item.status == ItemStatus.PENDING:
                    remaining.append(item)
            remaining.extend(self._queue)
            self._queue = remaining
            if not self._queue:
                self._new_item_event.clear()
            else:
                self._new_item_event.set()
        return collected

    def abort_wait(self) -> None:
        """Unblock any coroutine waiting in next_pending_batch_window (for shutdown)."""
        self._shutting_down = True
        self._new_item_event.set()

    async def mark_processing(self, item: QueueItem) -> None:
        async with self._lock:
            item.status = ItemStatus.PROCESSING
        await self._persist(item)

    async def mark_completed(self, item: QueueItem, image_name: str) -> None:
        async with self._lock:
            item.status = ItemStatus.COMPLETED
            item.output_image_name = image_name
        await self._persist(item)

    async def mark_failed(self, item: QueueItem, error: str) -> None:
        async with self._lock:
            item.status = ItemStatus.FAILED
            item.error = error
        await self._persist(item)

    async def mark_cancelled(self, item: QueueItem) -> None:
        """Mark a single item as CANCELLED (used by the worker on abort)."""
        async with self._lock:
            item.status = ItemStatus.CANCELLED
        await self._persist(item)

    async def cancel_processing_items(self) -> list[int]:
        """Mark all PROCESSING items as CANCELLED.  Returns their item_ids."""
        cancelled: list[int] = []
        async with self._lock:
            for item in self._all_items.values():
                if item.status == ItemStatus.PROCESSING:
                    item.status = ItemStatus.CANCELLED
                    cancelled.append(item.item_id)
        for item_id in cancelled:
            try:
                item = self._all_items.get(item_id)
                if item:
                    await self._persist(item)
            except Exception:
                pass
        return cancelled

    async def cancel_item(self, item_id: int) -> bool:
        """Cancel a pending item. Returns False if not pending or not found."""
        async with self._lock:
            item = self._all_items.get(item_id)
            if item is None or item.status != ItemStatus.PENDING:
                return False
            item.status = ItemStatus.CANCELLED
        await self._persist(item)
        return True

    async def cancel_all(self) -> list[QueueItem]:
        """Cancel all pending items. Returns the cancelled items."""
        cancelled: list[QueueItem] = []
        async with self._lock:
            for item in self._all_items.values():
                if item.status == ItemStatus.PENDING:
                    item.status = ItemStatus.CANCELLED
                    cancelled.append(item)
            self._queue.clear()
        for item in cancelled:
            await self._persist(item)
        return cancelled

    async def prune(self) -> int:
        """Remove completed/failed/cancelled items. Returns count pruned."""
        terminal = {ItemStatus.COMPLETED, ItemStatus.FAILED, ItemStatus.CANCELLED}
        to_remove: list[int] = []
        async with self._lock:
            to_remove = [
                item_id
                for item_id, item in self._all_items.items()
                if item.status in terminal
            ]
            for item_id in to_remove:
                del self._all_items[item_id]
        if self._store and to_remove:
            try:
                await self._store.delete_many(to_remove)
            except Exception:
                logger.warning("Failed to prune queue items from DB", exc_info=True)
        return len(to_remove)

    def get_item(self, item_id: int) -> QueueItem | None:
        return self._all_items.get(item_id)

    def all_items(self) -> list[QueueItem]:
        return list(self._all_items.values())
