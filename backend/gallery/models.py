from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
)
from sqlalchemy.orm import relationship

from backend.gallery.db import Base

# Many-to-many join table for images ↔ tags
image_tags = Table(
    "image_tags",
    Base.metadata,
    Column("image_id", Integer, ForeignKey("images.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)


class Gallery(Base):
    __tablename__ = "galleries"

    id = Column(Integer, primary_key=True)
    board_id = Column(String, unique=True, index=True)  # UUID string
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    images = relationship("ImageRecord", back_populates="gallery")


class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    image_name = Column(String, unique=True, index=True)  # filename e.g. "abc123.png"
    gallery_id = Column(Integer, ForeignKey("galleries.id"), nullable=True)
    gallery = relationship("Gallery", back_populates="images")

    # Generation metadata (denormalised for fast search)
    prompt = Column(String, default="")
    negative_prompt = Column(String, default="")
    seed = Column(Integer, default=0)
    width = Column(Integer, default=512)
    height = Column(Integer, default=512)
    steps = Column(Integer, default=20)
    cfg = Column(Float, default=7.5)
    cfg_rescale_multiplier = Column(Float, default=0.0)
    sampler = Column(String, default="")
    prediction_type = Column(String, default="default")
    model_key = Column(String, default="")
    model_name = Column(String, default="")

    # Monotonically increasing enqueue order (= QueueItem.item_id).
    # NULL for images created before this column was added.
    gen_seq = Column(Integer, nullable=True, index=True)

    # Img2img / source image metadata (nullable for backward compat)
    source_image = Column(String, nullable=True)
    source_width = Column(Integer, nullable=True)
    source_height = Column(Integer, nullable=True)
    strength = Column(Float, default=0.75)
    crop_left = Column(Float, default=0.0)
    crop_top = Column(Float, default=0.0)
    crop_width = Column(Float, default=0.0)
    crop_height = Column(Float, default=0.0)

    starred = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    tags = relationship("Tag", secondary=image_tags, back_populates="images")


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    images = relationship("ImageRecord", secondary=image_tags, back_populates="tags")

