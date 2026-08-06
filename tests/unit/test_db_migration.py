"""Unit tests for backend/gallery/db.py inline migrations."""
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from backend.gallery.db import _apply_inline_migrations


@pytest.mark.asyncio
async def test_inline_migration_renames_gen_seq_to_request_serial_number():
    """A legacy DB with a gen_seq column is upgraded in place: the column is
    renamed to request_serial_number and its values are preserved."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.execute(text(
            "CREATE TABLE images (id INTEGER PRIMARY KEY, image_name VARCHAR, gen_seq INTEGER)"
        ))
        await conn.execute(text(
            "INSERT INTO images (image_name, gen_seq) VALUES ('old.png', 7)"
        ))

        await conn.run_sync(_apply_inline_migrations)

        cols = await conn.run_sync(
            lambda sync_conn: {c["name"] for c in inspect(sync_conn).get_columns("images")}
        )
        assert "gen_seq" not in cols
        assert "request_serial_number" in cols

        row = (await conn.execute(text(
            "SELECT request_serial_number FROM images WHERE image_name = 'old.png'"
        ))).scalar_one()
        assert row == 7

        # Idempotent — running again must not raise.
        await conn.run_sync(_apply_inline_migrations)
    await engine.dispose()


@pytest.mark.asyncio
async def test_inline_migration_adds_request_serial_number_to_fresh_db():
    """A DB that already has request_serial_number (or no serial column at all)
    is left alone / gets the column added without error."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.execute(text(
            "CREATE TABLE images (id INTEGER PRIMARY KEY, image_name VARCHAR)"
        ))
        await conn.run_sync(_apply_inline_migrations)
        cols = await conn.run_sync(
            lambda sync_conn: {c["name"] for c in inspect(sync_conn).get_columns("images")}
        )
        assert "request_serial_number" in cols
        assert "gen_seq" not in cols
    await engine.dispose()
