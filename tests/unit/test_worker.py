"""Unit tests for PipelineWorker — submission serial preservation."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from backend.gallery.db import Base
import backend.gallery.models  # noqa: F401
from backend.gallery.service import GalleryService
from backend.domain import QueueItem, make_params
from backend.pipeline.worker import PipelineWorker


@pytest.fixture
async def worker_ctx():
    """A PipelineWorker wired to an in-memory gallery DB, plus the board id."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        svc = GalleryService(session)
        g = await svc.create_gallery("Default")
        await session.commit()
        board_id = g.board_id

    broadcaster = MagicMock()
    broadcaster.broadcast_event = AsyncMock()
    worker = PipelineWorker(
        queue=MagicMock(),
        registry=MagicMock(),
        loader=MagicMock(),
        broadcaster=broadcaster,
        session_factory=session_factory,
        default_board_id=board_id,
    )
    yield session_factory, board_id, broadcaster, worker
    await engine.dispose()


@pytest.mark.asyncio
async def test_record_in_gallery_preserves_submission_serial(worker_ctx):
    """The gallery record and the image_created WS event must both carry the
    submission serial (QueueItem.item_id), constant from enqueue time."""
    session_factory, board_id, broadcaster, worker = worker_ctx

    item = QueueItem(params=make_params())
    await worker._record_in_gallery(item, "img.png", "My Model")

    async with session_factory() as session:
        svc = GalleryService(session)
        records = await svc.list_images(board_id=board_id)
        assert len(records) == 1
        assert records[0].request_serial_number == item.item_id

    broadcaster.broadcast_event.assert_awaited_once()
    event_type, data = broadcaster.broadcast_event.await_args.args
    assert event_type == "image_created"
    assert data["request_serial_number"] == item.item_id
