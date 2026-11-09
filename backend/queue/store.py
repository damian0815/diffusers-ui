"""Async persistence store for queue items."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable

from sqlalchemy import delete as sql_delete, select, update as sql_update

from backend.domain import ImageGenerationParams, ItemStatus, QueueItem
from backend.queue.models import PersistedQueueItem

logger = logging.getLogger(__name__)


class QueueStore:
    """Persists QueueItem state to the SQLite database."""

    def __init__(self, session_factory: Callable):
        self._session_factory = session_factory

    async def upsert(self, item: QueueItem) -> None:
        """Insert a new item or update an existing one (status / outputs)."""
        async with self._session_factory() as session:
            stmt = select(PersistedQueueItem).where(
                PersistedQueueItem.item_id == item.item_id
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            now = datetime.utcnow()

            if row is None:
                row = PersistedQueueItem(
                    item_id=item.item_id,
                    queue_name="default",
                    status=item.status.value,
                    prompt=item.params.prompt,
                    negative_prompt=item.params.negative_prompt,
                    seed=item.params.seed,
                    width=item.params.width,
                    height=item.params.height,
                    steps=item.params.steps,
                    cfg=item.params.cfg,
                    cfg_rescale_multiplier=item.params.cfg_rescale_multiplier,
                    sampler=item.params.sampler,
                    model_key=item.params.model_key,
                    board_id=item.params.board_id,
                    prediction_type=item.params.prediction_type,
                    output_image_name=item.output_image_name,
                    error=item.error,
                    priority_value=item.priority_value,
                    lora_keys=json.dumps(item.params.lora_keys) if item.params.lora_keys else None,
                    lora_strengths=json.dumps(item.params.lora_strengths) if item.params.lora_strengths else None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.status = item.status.value
                row.output_image_name = item.output_image_name
                row.error = item.error
                row.priority_value = item.priority_value
                row.updated_at = now

            await session.commit()

    async def delete_many(self, item_ids: list[int]) -> None:
        """Remove a set of items from the database (used by prune)."""
        if not item_ids:
            return
        async with self._session_factory() as session:
            await session.execute(
                sql_delete(PersistedQueueItem).where(
                    PersistedQueueItem.item_id.in_(item_ids)
                )
            )
            await session.commit()

    async def reset_interrupted(self) -> None:
        """On startup: reset any 'processing' item to 'pending' so it reruns."""
        async with self._session_factory() as session:
            await session.execute(
                sql_update(PersistedQueueItem)
                .where(PersistedQueueItem.status == ItemStatus.PROCESSING.value)
                .values(status=ItemStatus.PENDING.value, updated_at=datetime.utcnow())
            )
            await session.commit()

    async def load_all(self) -> list[QueueItem]:
        """Load every tracked item in insertion order."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(PersistedQueueItem).order_by(PersistedQueueItem.id)
            )
            rows = result.scalars().all()
            return [_row_to_domain(row) for row in rows]


def _row_to_domain(row: PersistedQueueItem) -> QueueItem:
    """Reconstruct a QueueItem from a DB row without triggering _next_id()."""
    params = ImageGenerationParams(
        prompt=row.prompt,
        negative_prompt=row.negative_prompt,
        seed=row.seed,
        width=row.width,
        height=row.height,
        steps=row.steps,
        cfg=row.cfg,
        cfg_rescale_multiplier=row.cfg_rescale_multiplier,
        sampler=row.sampler,
        model_key=row.model_key,
        board_id=row.board_id,
        prediction_type=row.prediction_type if row.prediction_type else "default",
        lora_keys=json.loads(row.lora_keys) if row.lora_keys else [],
        lora_strengths=json.loads(row.lora_strengths) if row.lora_strengths else [],
    )
    # Pass item_id explicitly so default_factory (_next_id) is NOT called.
    # The legacy ``priority`` (run-now) column is ignored: ordering now comes
    # from priority_value alone.  Rows from an older DB that were enqueued with
    # priority=True but rank 1 keep rank 1; their original queue position is
    # preserved by the ordered restore in main.py.
    return QueueItem(
        params=params,
        item_id=row.item_id,
        status=ItemStatus(row.status),
        output_image_name=row.output_image_name,
        error=row.error,
        priority_value=row.priority_value if row.priority_value is not None else 1,
        created_at=row.created_at,
    )