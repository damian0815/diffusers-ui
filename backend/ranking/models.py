"""Persistence layer for the Ranking Studio (RLHF preference data) feature.

These models share the single project-wide ``Base`` from
``backend.gallery.db`` so ``init_db()`` creates them at startup alongside the
gallery tables. Column additions *after* first release must go through
``backend.gallery/db.py:_apply_inline_migrations()`` — ``create_all`` never
alters existing tables.

Image-name referential integrity (decision documented per issue):

* ``ranking_hands`` stores image names inside JSON arrays, which cannot carry
  SQL foreign keys. A hand is an immutable historical record: deleting a
  gallery image does NOT modify or remove hand rows. Downstream consumers
  (trainer, export, sampler) join names against the live ``images`` table and
  simply skip names that no longer exist.
* ``image_embeddings`` declares ``ON DELETE CASCADE`` on its FK to
  ``images.image_name``: cache rows are disposable — they can always be
  recomputed from the image file. Note that SQLite only enforces foreign
  keys when ``PRAGMA foreign_keys=ON``, which this app does not set (changing
  that would affect every table), so in practice stale cache rows for deleted
  images are *ignored* rather than cascaded away: every read goes through
  ``RankingStore.get_embeddings(image_names, model_tag)`` with names sourced
  from the live gallery, so unreachable rows are never returned, and a
  re-embed of a reused name overwrites the row via upsert.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
)

from backend.gallery.db import Base


class RankingHand(Base):
    """One SUBMITTED ranking hand (best-to-worst ordering of same-request
    image variants), i.e. one accumulated listwise preference label.

    ``hand_index`` is the 0-based submission order and the public addressing
    scheme for hand history ("backtrack to hand 3"); ``id`` is only the
    internal surrogate key (mirrors ``PersistedQueueItem``).
    """

    __tablename__ = "ranking_hands"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 0-based submission order. Unique: two hands may never share an address.
    hand_index = Column(Integer, unique=True, index=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # JSON array of ``images.image_name`` strings, ordered best first.
    ranked_images = Column(JSON, nullable=False)

    # JSON array of ``images.image_name`` strings — cards skipped as
    # irrelevant ("abstain" signal for export). Order carries no meaning.
    discarded_images = Column(JSON, nullable=False, default=list)

    # The seed_id active when this hand was drawn (up to 5 pinned images
    # became leading cards). Persisted so the exact hand can be reproduced.
    sampler_seed = Column(Integer, nullable=False)


class ImageEmbedding(Base):
    """Frozen per-image embedding cache for dedupe / cluster-spread.

    One row per image (``image_name`` is the PK). ``embedding`` is the raw
    float32 vector packed as a native-endian BLOB — pack/unpack helpers
    live in ``backend.ranking.store`` (``_pack_vector`` / ``_unpack_vector``).

    ``model_tag`` identifies which embedding model produced the vector. When
    the model changes, rows with a stale tag are IGNORED on read (never
    deleted); the next embed pass overwrites them via upsert.
    """

    __tablename__ = "image_embeddings"

    image_name = Column(
        String,
        ForeignKey("images.image_name", ondelete="CASCADE"),
        primary_key=True,
    )

    embedding = Column(LargeBinary, nullable=False)
    model_tag = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class RankingState(Base):
    """Key/value table for scalar Ranking Studio state.

    One row per well-known key (rather than one wide row) so new persisted
    fields never need a schema migration. Access only through
    ``RankingStore.get_state`` / ``set_state``; values are stored as TEXT,
    with typed helpers in the store. Known keys are defined as constants in
    ``backend.ranking.store``:

    * ``sampler_rng_seed``  — int; persisted RNG seed so hand generation is
      reproducible across restarts.
    * ``last_retrain_at``   — ISO-8601 timestamp of the most recent ranker
      retrain.
    * ``ranker_checkpoint`` — path to the ranker checkpoint file.
    * ``ranker_version``    — checkpoint/model version string.
    """

    __tablename__ = "ranking_state"

    key = Column(String, primary_key=True)
    value = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
