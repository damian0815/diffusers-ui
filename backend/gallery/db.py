from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.db_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


def _apply_inline_migrations(conn) -> None:
    """Idempotent schema additions introduced after the initial release."""
    from sqlalchemy import inspect, text
    inspector = inspect(conn)
    tables = inspector.get_table_names()
    if "images" in tables:
        existing = {c["name"] for c in inspector.get_columns("images")}
        if "gen_seq" not in existing:
            conn.execute(text("ALTER TABLE images ADD COLUMN gen_seq INTEGER"))
        if "prediction_type" not in existing:
            conn.execute(text("ALTER TABLE images ADD COLUMN prediction_type VARCHAR DEFAULT 'default'"))
        for col, coltype in [
            ("source_image", "VARCHAR"),
            ("source_width", "INTEGER"),
            ("source_height", "INTEGER"),
            ("strength", "FLOAT DEFAULT 0.75"),
            ("crop_left", "FLOAT DEFAULT 0.0"),
            ("crop_top", "FLOAT DEFAULT 0.0"),
            ("crop_width", "FLOAT DEFAULT 0.0"),
            ("crop_height", "FLOAT DEFAULT 0.0"),
        ]:
            if col not in existing:
                conn.execute(text(f"ALTER TABLE images ADD COLUMN {col} {coltype}"))
        # Add lora_keys and lora_strengths columns
        if "lora_keys" not in existing:
            conn.execute(text("ALTER TABLE images ADD COLUMN lora_keys TEXT"))
            conn.execute(text("ALTER TABLE images ADD COLUMN lora_strengths TEXT"))
    if "queue_items" in tables:
        existing = {c["name"] for c in inspector.get_columns("queue_items")}
        if "board_id" not in existing:
            conn.execute(text("ALTER TABLE queue_items ADD COLUMN board_id VARCHAR"))
        if "prediction_type" not in existing:
            conn.execute(text("ALTER TABLE queue_items ADD COLUMN prediction_type VARCHAR NOT NULL DEFAULT 'default'"))
        if "lora_keys" not in existing:
            conn.execute(text("ALTER TABLE queue_items ADD COLUMN lora_keys TEXT"))
            conn.execute(text("ALTER TABLE queue_items ADD COLUMN lora_strengths TEXT"))


async def init_db() -> None:
    """Create all tables. Called once at startup."""
    # Import all models so their metadata is registered before create_all
    import backend.gallery.models  # noqa: F401
    import backend.queue.models    # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_apply_inline_migrations)


async def get_session() -> AsyncSession:
    """FastAPI dependency. Yields a session and commits/rolls back."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

