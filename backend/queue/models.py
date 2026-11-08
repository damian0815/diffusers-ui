"""SQLAlchemy model that persists queue items across server restarts."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from backend.gallery.db import Base


class PersistedQueueItem(Base):
    __tablename__ = "queue_items"

    # Internal DB surrogate key (preserves insertion order)
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Domain-level item identity (matches QueueItem.item_id)
    item_id = Column(Integer, unique=True, nullable=False, index=True)

    queue_name = Column(String, nullable=False, default="default")
    status = Column(String, nullable=False)  # ItemStatus string value

    # Generation params (denormalised for easy restore)
    prompt = Column(String, nullable=False, default="")
    negative_prompt = Column(String, nullable=False, default="")
    seed = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    steps = Column(Integer, nullable=False)
    cfg = Column(Float, nullable=False)
    cfg_rescale_multiplier = Column(Float, nullable=False, default=0.0)
    sampler = Column(String, nullable=False)
    model_key = Column(String, nullable=False)
    board_id = Column(String, nullable=True)           # target gallery board
    prediction_type = Column(String, nullable=False, default="default")

    output_image_name = Column(String, nullable=True)
    error = Column(String, nullable=True)

    priority = Column(Boolean, nullable=False, default=False)
    # Ordering rank (higher = earlier); does not affect the run-now block.
    priority_value = Column(Integer, nullable=False, default=1)

    # LoRA metadata (JSON-encoded lists for backward compat)
    lora_keys = Column(String, nullable=True)       # JSON-encoded list of filenames
    lora_strengths = Column(String, nullable=True)   # JSON-encoded list of strengths

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

