"""Integration tests for QueueStore — restart persistence of priority + created_at."""
import pytest
from datetime import datetime
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from backend.domain import QueueItem, make_params


async def _make_session_factory(tmp_path):
    db_path = tmp_path / "queue_store.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory


@pytest.mark.asyncio
async def test_store_round_trips_created_at(tmp_path):
    """Every item round-trips with a created_at timestamp and a default
    priority_value of 1; status updates round-trip too."""
    import backend.queue.models  # noqa: F401 — register ORM models
    from backend.gallery.db import Base
    from backend.queue.store import QueueStore

    engine, session_factory = await _make_session_factory(tmp_path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    store = QueueStore(session_factory)

    boosted = QueueItem(params=make_params(), priority_value=2)
    await store.upsert(boosted)

    normal = QueueItem(params=make_params())
    await store.upsert(normal)

    items = await store.load_all()
    assert len(items) == 2

    by_id = {it.item_id: it for it in items}
    assert by_id[boosted.item_id].priority_value == 2
    assert by_id[normal.item_id].priority_value == 1

    created = by_id[boosted.item_id].created_at
    assert created is not None
    age_s = (datetime.utcnow() - created).total_seconds()
    assert 0 <= age_s < 60

    await engine.dispose()


@pytest.mark.asyncio
async def test_store_round_trips_priority_value(tmp_path):
    """priority_value persists (and defaults to 1); later updates round-trip."""
    import backend.queue.models  # noqa: F401 — register ORM models
    from backend.gallery.db import Base
    from backend.queue.store import QueueStore

    engine, session_factory = await _make_session_factory(tmp_path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    store = QueueStore(session_factory)

    ranked = QueueItem(params=make_params(), priority_value=5)
    await store.upsert(ranked)

    normal = QueueItem(params=make_params())
    await store.upsert(normal)

    items = await store.load_all()
    by_id = {it.item_id: it for it in items}
    assert by_id[ranked.item_id].priority_value == 5
    assert by_id[normal.item_id].priority_value == 1

    # The update branch of upsert must persist rank changes too.
    ranked.priority_value = 2
    await store.upsert(ranked)
    items = await store.load_all()
    by_id = {it.item_id: it for it in items}
    assert by_id[ranked.item_id].priority_value == 2

    await engine.dispose()


@pytest.mark.asyncio
async def test_inline_migration_adds_priority_to_queue_items(tmp_path):
    """A DB whose queue_items table predates the priority column is migrated in
    place; existing rows read back as non-priority."""
    from backend.gallery.db import _apply_inline_migrations

    engine, _ = await _make_session_factory(tmp_path)
    async with engine.begin() as conn:
        await conn.execute(text(
            "CREATE TABLE queue_items ("
            " id INTEGER PRIMARY KEY,"
            " item_id INTEGER UNIQUE NOT NULL,"
            " queue_name VARCHAR NOT NULL DEFAULT 'default',"
            " status VARCHAR NOT NULL,"
            " prompt VARCHAR NOT NULL DEFAULT '',"
            " negative_prompt VARCHAR NOT NULL DEFAULT '',"
            " seed INTEGER NOT NULL,"
            " width INTEGER NOT NULL,"
            " height INTEGER NOT NULL,"
            " steps INTEGER NOT NULL,"
            " cfg FLOAT NOT NULL,"
            " cfg_rescale_multiplier FLOAT NOT NULL DEFAULT 0.0,"
            " sampler VARCHAR NOT NULL,"
            " model_key VARCHAR NOT NULL,"
            " board_id VARCHAR,"
            " prediction_type VARCHAR NOT NULL DEFAULT 'default',"
            " output_image_name VARCHAR,"
            " error VARCHAR,"
            " created_at DATETIME NOT NULL,"
            " updated_at DATETIME NOT NULL)"
        ))
        await conn.execute(text(
            "INSERT INTO queue_items (item_id, status, seed, width, height, steps, cfg, sampler, model_key, created_at, updated_at) "
            "VALUES (1, 'pending', 1, 512, 512, 20, 7.5, 'euler_a', 'm', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
        ))

        await conn.run_sync(_apply_inline_migrations)

        cols = await conn.run_sync(
            lambda sync_conn: {c["name"] for c in inspect(sync_conn).get_columns("queue_items")}
        )
        assert "priority" in cols

        row = (await conn.execute(text(
            "SELECT priority FROM queue_items WHERE item_id = 1"
        ))).scalar_one()
        assert row == 0

        # priority_value is added alongside; pre-existing rows read the default.
        assert "priority_value" in cols
        rank_row = (await conn.execute(text(
            "SELECT priority_value FROM queue_items WHERE item_id = 1"
        ))).scalar_one()
        assert rank_row == 1

        # Idempotent — running again must not raise.
        await conn.run_sync(_apply_inline_migrations)
    await engine.dispose()


@pytest.mark.asyncio
async def test_legacy_priority_rows_load_as_ranked_items(tmp_path):
    """DB-compat: rows persisted before the run-now boolean was removed
    (priority=1) still load — the legacy column is simply ignored."""
    import backend.queue.models  # noqa: F401 — register ORM models
    from backend.gallery.db import Base
    from backend.queue.store import QueueStore

    engine, session_factory = await _make_session_factory(tmp_path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    store = QueueStore(session_factory)
    item = QueueItem(params=make_params(), priority_value=3)
    await store.upsert(item)
    # Force the raw legacy flag onto the stored row (upsert no longer writes it).
    async with engine.begin() as conn:
        await conn.execute(text(
            "UPDATE queue_items SET priority = 1 WHERE item_id = :iid"
        ), {"iid": item.item_id})

    loaded = {it.item_id: it for it in await store.load_all()}
    assert loaded[item.item_id].priority_value == 3
    assert not hasattr(loaded[item.item_id], "priority")
    await engine.dispose()
