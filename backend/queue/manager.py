import asyncio
from collections import deque

from backend.domain import QueueItem, ItemStatus

QUEUE_POLL_INTERVAL_S = 0.05


class QueueManager:
    """
    Manages a queue of QueueItems.

    Responsibilities:
    - Accepting new items (append or prepend)
    - Providing the next pending item batch to the worker
    - Tracking status of all items (pending → processing → completed/failed/cancelled)
    - Cancellation of individual items or the entire queue
    """

    def __init__(self):
        self._queue: deque[QueueItem] = deque()
        self._all_items: dict[int, QueueItem] = {}
        self._lock = asyncio.Lock()
        self._new_item_event = asyncio.Event()

    async def enqueue(self, item: QueueItem, prepend: bool = False) -> QueueItem:
        """Add an item to the queue. prepend=True puts it at the front."""
        async with self._lock:
            if prepend:
                self._queue.appendleft(item)
            else:
                self._queue.append(item)
            self._all_items[item.item_id] = item
        self._new_item_event.set()
        return item

    async def next_pending_batch_window(self, window_s: float) -> list[QueueItem]:
        """
        Wait until at least one pending item is available, then collect all items
        that arrive within `window_s` seconds. Returns the collected items,
        removing them from the queue. Items cancelled during the window are excluded.
        """
        # Wait for at least one item
        await self._new_item_event.wait()

        # Collect items over the window
        if window_s > 0:
            await asyncio.sleep(window_s)

        async with self._lock:
            self._new_item_event.clear()
            collected: list[QueueItem] = []
            while self._queue:
                item = self._queue.popleft()
                if item.status == ItemStatus.PENDING:
                    collected.append(item)
                # Skip cancelled items — they stay in _all_items for status tracking
            # If nothing was collected but the event was set, it might have been all
            # cancelled; reset event only if queue is still empty
            if not self._queue:
                self._new_item_event.clear()
            else:
                self._new_item_event.set()
        return collected

    async def mark_processing(self, item: QueueItem) -> None:
        """Transition item to PROCESSING status."""
        async with self._lock:
            item.status = ItemStatus.PROCESSING

    async def mark_completed(self, item: QueueItem, image_name: str) -> None:
        """Transition item to COMPLETED and record the output filename."""
        async with self._lock:
            item.status = ItemStatus.COMPLETED
            item.output_image_name = image_name

    async def mark_failed(self, item: QueueItem, error: str) -> None:
        """Transition item to FAILED and record the error message."""
        async with self._lock:
            item.status = ItemStatus.FAILED
            item.error = error

    async def cancel_item(self, item_id: int) -> bool:
        """
        Cancel a pending item. Returns False if item is already processing or
        does not exist.
        """
        async with self._lock:
            item = self._all_items.get(item_id)
            if item is None or item.status != ItemStatus.PENDING:
                return False
            item.status = ItemStatus.CANCELLED
        return True

    async def cancel_all(self) -> int:
        """Cancel all pending items. Returns count cancelled."""
        count = 0
        async with self._lock:
            for item in self._all_items.values():
                if item.status == ItemStatus.PENDING:
                    item.status = ItemStatus.CANCELLED
                    count += 1
            self._queue.clear()
        return count

    async def prune(self) -> int:
        """Remove completed/failed/cancelled items from tracking. Returns count pruned."""
        terminal = {ItemStatus.COMPLETED, ItemStatus.FAILED, ItemStatus.CANCELLED}
        to_remove = [
            item_id
            for item_id, item in self._all_items.items()
            if item.status in terminal
        ]
        async with self._lock:
            for item_id in to_remove:
                del self._all_items[item_id]
        return len(to_remove)

    def get_item(self, item_id: int) -> QueueItem | None:
        return self._all_items.get(item_id)

    def all_items(self) -> list[QueueItem]:
        return list(self._all_items.values())

