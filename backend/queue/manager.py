import asyncio
import logging
from collections import deque

from backend.domain import QueueItem, ItemStatus

logger = logging.getLogger(__name__)

QUEUE_POLL_INTERVAL_S = 0.05


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

    async def enqueue(self, item: QueueItem, prepend: bool = False) -> QueueItem:
        """Add an item to the queue. prepend=True puts it at the front."""
        async with self._lock:
            if prepend:
                self._queue.appendleft(item)
            else:
                self._queue.append(item)
            self._all_items[item.item_id] = item
        await self._persist(item)
        self._new_item_event.set()
        return item

    async def requeue_items(self, items: list[QueueItem], prepend: bool = True) -> None:
        """
        Reset *items* to PENDING and put them back in the queue.

        Used after an OOM failure to retry at a smaller batch size.
        When *prepend* is True (default) items are placed at the front so they
        are picked up before any subsequently-enqueued work.
        """
        async with self._lock:
            ordered = list(reversed(items)) if prepend else items
            for item in ordered:
                item.status = ItemStatus.PENDING
                item.error = None
                if prepend:
                    self._queue.appendleft(item)
                else:
                    self._queue.append(item)
        for item in items:
            await self._persist(item)
        self._new_item_event.set()

    async def move_to_front(self, item_id: int) -> bool:
        """
        Move a *pending* item to the front of the queue.

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
            self._queue.appendleft(item)
        return True

    async def next_pending_batch_window(self, window_s: float) -> list[QueueItem]:
        """
        Wait until at least one pending item is available, then collect all
        items that arrive within ``window_s`` seconds.  Returns an empty list
        if ``abort_wait()`` was called (shutdown signal).
        """
        await self._new_item_event.wait()

        if self._shutting_down:
            return []

        if window_s > 0:
            await asyncio.sleep(window_s)

        async with self._lock:
            self._new_item_event.clear()
            collected: list[QueueItem] = []
            while self._queue:
                item = self._queue.popleft()
                if item.status == ItemStatus.PENDING:
                    collected.append(item)
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

    async def cancel_item(self, item_id: int) -> bool:
        """Cancel a pending item. Returns False if not pending or not found."""
        async with self._lock:
            item = self._all_items.get(item_id)
            if item is None or item.status != ItemStatus.PENDING:
                return False
            item.status = ItemStatus.CANCELLED
        await self._persist(item)
        return True

    async def cancel_all(self) -> int:
        """Cancel all pending items. Returns count cancelled."""
        count = 0
        cancelled: list[QueueItem] = []
        async with self._lock:
            for item in self._all_items.values():
                if item.status == ItemStatus.PENDING:
                    item.status = ItemStatus.CANCELLED
                    cancelled.append(item)
                    count += 1
            self._queue.clear()
        for item in cancelled:
            await self._persist(item)
        return count

    async def prune(self) -> int:
        """Remove completed/failed/cancelled items. Returns count pruned."""
        terminal = {ItemStatus.COMPLETED, ItemStatus.FAILED, ItemStatus.CANCELLED}
        to_remove = [
            item_id
            for item_id, item in self._all_items.items()
            if item.status in terminal
        ]
        async with self._lock:
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
