"""Integration tests for the Ranking Studio persistence layer (RankingStore)."""
import asyncio

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from backend.ranking.store import (
    KEY_SAMPLER_RNG_SEED,
    RankingStore,
    SubmittedHand,
)


async def _make_store(tmp_path):
    """Fresh file-based SQLite DB + RankingStore over the full gallery schema."""
    import backend.gallery.models  # noqa: F401 — register ORM models
    import backend.ranking.models  # noqa: F401
    from backend.gallery.db import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ranking.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory, RankingStore(session_factory)


@pytest.mark.asyncio
async def test_init_db_creates_ranking_tables(tmp_path, monkeypatch):
    """init_db() imports backend.ranking.models and create_all()s them on the
    shared gallery Base — the exact startup wiring this issue requires."""
    from backend.gallery.db import Base
    import backend.gallery.db as db_module

    test_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'startup.db'}")
    monkeypatch.setattr(db_module, "engine", test_engine)
    await db_module.init_db()

    async with test_engine.begin() as conn:
        tables = await conn.run_sync(
            lambda sync_conn: set(inspect(sync_conn).get_table_names())
        )
    assert {"ranking_hands", "image_embeddings", "ranking_state"} <= tables
    # Same Base as the gallery tables (required for create_all in one pass)
    assert {"images", "galleries", "tags"} & set(Base.metadata.tables)

    await test_engine.dispose()


@pytest.mark.asyncio
async def test_append_hand_assigns_sequential_indices(tmp_path):
    engine, _, store = await _make_store(tmp_path)

    first = await store.append_hand(["a.png", "b.png"], ["c.png"], sampler_seed=7)
    second = await store.append_hand(["d.png", "e.png"], [], sampler_seed=8)

    assert isinstance(first, SubmittedHand)
    assert first.hand_index == 0
    assert second.hand_index == 1
    assert first.ranked_images == ["a.png", "b.png"]
    assert first.discarded_images == ["c.png"]
    assert first.sampler_seed == 7
    assert first.created_at is not None

    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_appends_do_not_collide(tmp_path):
    """append_hand serialises index assignment under its internal lock."""
    engine, _, store = await _make_store(tmp_path)

    hands = await asyncio.gather(
        store.append_hand(["w.png"], [], sampler_seed=1),
        store.append_hand(["x.png"], [], sampler_seed=2),
        store.append_hand(["y.png"], [], sampler_seed=3),
    )
    assert sorted(h.hand_index for h in hands) == [0, 1, 2]

    await engine.dispose()


@pytest.mark.asyncio
async def test_hand_history_queries(tmp_path):
    engine, _, store = await _make_store(tmp_path)

    await store.append_hand(["a.png", "b.png"], ["x.png"], sampler_seed=1)
    await store.append_hand(["c.png", "d.png"], ["y.png", "x.png"], sampler_seed=2)

    assert (await store.get_hand_by_index(1)).ranked_images == ["c.png", "d.png"]
    assert await store.get_hand_by_index(5) is None

    hands = await store.list_all_hands()
    assert [h.hand_index for h in hands] == [0, 1]
    assert await store.all_hands_flat_order() == [
        ["a.png", "b.png"],
        ["c.png", "d.png"],
    ]

    assert await store.get_ranked_names() == {"a.png", "b.png", "c.png", "d.png"}
    # discard union de-duplicates across hands
    assert await store.get_discarded_names() == {"x.png", "y.png"}

    await engine.dispose()


@pytest.mark.asyncio
async def test_embedding_cache_round_trip_and_stale_tag(tmp_path):
    engine, _, store = await _make_store(tmp_path)

    vector = [0.25, -1.5, 3.0]
    await store.add_embedding("a.png", "clip-vit-b32", vector)

    got = await store.get_embeddings(["a.png", "missing.png"], "clip-vit-b32")
    assert list(got) == ["a.png"]
    assert got["a.png"] == pytest.approx(vector)

    # A different model_tag is a cache MISS — stale rows are ignored, not deleted.
    assert await store.get_embeddings(["a.png"], "other-model") == {}

    # Re-embedding upserts (image_name is PK): the new tag wins, still one row.
    await store.add_embedding("a.png", "other-model", [9.0])
    assert await store.get_embeddings(["a.png"], "clip-vit-b32") == {}
    got = await store.get_embeddings(["a.png"], "other-model")
    assert got == {"a.png": pytest.approx([9.0])}

    assert await store.get_embeddings([], "other-model") == {}

    await engine.dispose()


@pytest.mark.asyncio
async def test_state_key_value_round_trip(tmp_path):
    engine, _, store = await _make_store(tmp_path)

    assert await store.get_state(KEY_SAMPLER_RNG_SEED) is None
    assert await store.get_state_int(KEY_SAMPLER_RNG_SEED, default=42) == 42

    await store.set_state(KEY_SAMPLER_RNG_SEED, "12345")
    assert await store.get_state_int(KEY_SAMPLER_RNG_SEED) == 12345

    # Upsert, then clear.
    await store.set_state(KEY_SAMPLER_RNG_SEED, "999")
    assert await store.get_state_int(KEY_SAMPLER_RNG_SEED) == 999
    await store.set_state(KEY_SAMPLER_RNG_SEED, None)
    assert await store.get_state_int(KEY_SAMPLER_RNG_SEED) is None

    await engine.dispose()


@pytest.mark.asyncio
async def test_deleting_image_leaves_hand_history_intact(tmp_path):
    """Hands are immutable records: an image name removed from the gallery
    still appears in stored hands (consumers skip unresolvable names)."""
    engine, session_factory, store = await _make_store(tmp_path)

    import backend.gallery.models  # noqa: F401
    from backend.gallery.models import ImageRecord

    async with session_factory() as session:
        session.add(ImageRecord(image_name="gone.png", prompt="p"))
        await session.commit()

    await store.append_hand(["gone.png", "stay.png"], [], sampler_seed=1)

    async with session_factory() as session:
        row = await session.scalar(
            select(ImageRecord).where(ImageRecord.image_name == "gone.png")
        )
        await session.delete(row)
        await session.commit()

    hand = await store.get_hand_by_index(0)
    assert hand.ranked_images == ["gone.png", "stay.png"]
    assert "gone.png" in await store.get_ranked_names()

    await engine.dispose()
