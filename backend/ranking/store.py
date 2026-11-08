"""Async persistence service for Ranking Studio data.

``RankingStore`` follows the ``QueueStore`` pattern: it is constructed with a
session *factory* (``AsyncSessionLocal``) and opens/closes its own session per
operation, so both request handlers and background tasks (trainer, sampler)
can share one instance. All DB access for the ranking tables goes through
this class — no SQL in callers.

The store returns plain ``SubmittedHand`` dataclasses instead of ORM rows so
callers never depend on ``expire_on_commit`` session configuration (the row
would raise when touched after its session closed).
"""
from __future__ import annotations

import asyncio
from array import array
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.ranking.models import ImageEmbedding, RankingHand, RankingState

# ---------------------------------------------------------------------------
# Well-known ranking_state keys (see RankingState docstring)
# ---------------------------------------------------------------------------
KEY_SAMPLER_RNG_SEED = "sampler_rng_seed"
KEY_LAST_RETRAIN_AT = "last_retrain_at"
KEY_RANKER_CHECKPOINT = "ranker_checkpoint"
KEY_RANKER_VERSION = "ranker_version"


@dataclass
class SubmittedHand:
    """A submitted ranking hand as read back from ``ranking_hands``."""

    hand_index: int
    ranked_images: list[str]        # ordered, best first
    discarded_images: list[str]
    sampler_seed: int
    created_at: datetime | None = None


def _pack_vector(vector: Sequence[float]) -> bytes:
    """Pack floats into a native-endian float32 BLOB.

    ``array('f')`` is exactly 4 bytes per element on every platform this app
    runs on; the same process family writes and reads the cache, so native
    byte order is fine (documented on ``ImageEmbedding``).
    """
    return array("f", vector).tobytes()


def _unpack_vector(blob: bytes) -> list[float]:
    """Inverse of ``_pack_vector``."""
    values = array("f")
    values.frombytes(blob)
    return list(values)


def _to_submitted_hand(row: RankingHand) -> SubmittedHand:
    """Snapshot an ORM row into a detached dataclass (call while the
    originating session is still open)."""
    return SubmittedHand(
        hand_index=row.hand_index,
        ranked_images=list(row.ranked_images or []),
        discarded_images=list(row.discarded_images or []),
        sampler_seed=row.sampler_seed,
        created_at=row.created_at,
    )


class RankingStore:
    """All database operations for hands, embedding cache, and state."""

    def __init__(self, session_factory: Callable):
        self._session_factory = session_factory
        # Serialises the read-max-then-insert in append_hand() so two
        # concurrent submissions can never claim the same hand_index.
        self._append_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Hands
    # ------------------------------------------------------------------

    async def append_hand(
        self,
        ranked_images: Sequence[str],
        discarded_images: Sequence[str],
        sampler_seed: int,
    ) -> SubmittedHand:
        """Record a submitted hand; assigns the next 0-based ``hand_index``."""
        async with self._append_lock:
            async with self._session_factory() as session:
                max_index = await session.scalar(
                    select(func.max(RankingHand.hand_index))
                )
                row = RankingHand(
                    hand_index=0 if max_index is None else max_index + 1,
                    ranked_images=list(ranked_images),
                    discarded_images=list(discarded_images),
                    sampler_seed=int(sampler_seed),
                )
                session.add(row)
                # flush() populates the server-side defaults (created_at);
                # snapshot before commit() so the result is safe even with
                # expire_on_commit=True session factories.
                await session.flush()
                hand = _to_submitted_hand(row)
                await session.commit()
                return hand

    async def get_hand_by_index(self, hand_index: int) -> SubmittedHand | None:
        """Index-addressable hand history; None if that hand was never submitted."""
        async with self._session_factory() as session:
            row = await session.scalar(
                select(RankingHand).where(RankingHand.hand_index == hand_index)
            )
            return _to_submitted_hand(row) if row is not None else None

    async def list_all_hands(self) -> list[SubmittedHand]:
        """Every submitted hand, oldest submission first."""
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(RankingHand).order_by(RankingHand.hand_index)
                )
            ).all()
            return [_to_submitted_hand(row) for row in rows]

    async def all_hands_flat_order(self) -> list[list[str]]:
        """Accumulated rankings for the trainer: each inner list is one
        hand's ``ranked_images`` (best first); the outer list is in
        submission order. Discards are not part of the ordered training
        signal — see ``get_discarded_names`` for the abstain set."""
        return [hand.ranked_images for hand in await self.list_all_hands()]

    async def get_ranked_names(self) -> set[str]:
        """Union of every image that has appeared in a submitted hand's
        ranked list. Callers use this to exclude annotated images from
        future candidate pools (exclusion logic itself lives elsewhere)."""
        async with self._session_factory() as session:
            rows = (await session.scalars(select(RankingHand.ranked_images))).all()
        names: set[str] = set()
        for ranked in rows:
            names.update(ranked or [])
        return names

    async def get_discarded_names(self) -> set[str]:
        """Union of every image discarded (skipped as irrelevant) across all
        submitted hands — the abstain signal / pool exclusion set."""
        async with self._session_factory() as session:
            rows = (await session.scalars(select(RankingHand.discarded_images))).all()
        names: set[str] = set()
        for discarded in rows:
            names.update(discarded or [])
        return names

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    async def add_embedding(
        self, image_name: str, model_tag: str, vector: Sequence[float]
    ) -> None:
        """Upsert the cached embedding for one image.

        ``image_name`` is the PK, so a re-embed (e.g. after the embedding
        model changes) overwrites the previous row rather than duplicating.
        """
        stmt = (
            sqlite_insert(ImageEmbedding)
            .values(
                image_name=image_name,
                embedding=_pack_vector(vector),
                model_tag=model_tag,
                created_at=datetime.utcnow(),
            )
            .on_conflict_do_update(
                index_elements=[ImageEmbedding.image_name],
                set_={
                    "embedding": _pack_vector(vector),
                    "model_tag": model_tag,
                    "created_at": datetime.utcnow(),
                },
            )
        )
        async with self._session_factory() as session:
            await session.execute(stmt)
            await session.commit()

    async def get_embeddings(
        self, image_names: Sequence[str], model_tag: str
    ) -> dict[str, list[float]]:
        """Cached vectors for *image_names* that were produced by *model_tag*.

        Rows with a stale ``model_tag`` are ignored (never deleted), so after
        an embedding-model change this returns a cache miss until the images
        are re-embedded. Missing names are simply absent from the result.
        """
        if not image_names:
            return {}
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(ImageEmbedding.image_name, ImageEmbedding.embedding).where(
                        ImageEmbedding.image_name.in_(list(image_names)),
                        ImageEmbedding.model_tag == model_tag,
                    )
                )
            ).all()
        return {name: _unpack_vector(blob) for name, blob in rows}

    # ------------------------------------------------------------------
    # Scalar state (ranking_state key/value table)
    # ------------------------------------------------------------------

    async def get_state(self, key: str) -> str | None:
        """Raw TEXT value for *key*, or None if never set."""
        async with self._session_factory() as session:
            return await session.scalar(
                select(RankingState.value).where(RankingState.key == key)
            )

    async def get_state_int(self, key: str, default: int | None = None) -> int | None:
        """Integer-typed state value (e.g. ``KEY_SAMPLER_RNG_SEED``)."""
        raw = await self.get_state(key)
        if raw is None or raw == "":
            return default
        return int(raw)

    async def set_state(self, key: str, value: str | None) -> None:
        """Upsert a state value; store None to clear it."""
        stmt = (
            sqlite_insert(RankingState)
            .values(key=key, value=value, updated_at=datetime.utcnow())
            .on_conflict_do_update(
                index_elements=[RankingState.key],
                set_={"value": value, "updated_at": datetime.utcnow()},
            )
        )
        async with self._session_factory() as session:
            await session.execute(stmt)
            await session.commit()
