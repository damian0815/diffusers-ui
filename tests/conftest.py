"""
Shared test fixtures used across unit and integration tests.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Helpers (importable from conftest or domain)
# ---------------------------------------------------------------------------

def make_params(**overrides):
    """Build a valid ImageGenerationParams with sensible defaults."""
    from backend.domain import ImageGenerationParams
    defaults = dict(
        prompt="test prompt",
        negative_prompt="",
        seed=42,
        width=512,
        height=512,
        steps=20,
        cfg=7.5,
        cfg_rescale_multiplier=0.0,
        sampler="euler_a",
        model_key="test_model",
    )
    return ImageGenerationParams(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pipeline():
    """A fake DiffusionPipeline that returns a white image instantly."""
    pipeline = MagicMock()
    pipeline.return_value.images = [PILImage.new("RGB", (512, 512), "white")]
    pipeline.scheduler = MagicMock()
    pipeline.scheduler.config = {}
    return pipeline


@pytest.fixture
async def client(mock_pipeline, tmp_path, monkeypatch):
    """
    Integration test client:
    - In-memory SQLite database
    - Mock pipeline loader (no GPU required)
    - Temporary images/thumbs directories
    """
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    from backend.gallery.db import Base
    import backend.gallery.models  # noqa: F401 — register ORM models

    # Patch settings paths
    from backend import config as cfg_mod
    monkeypatch.setattr(cfg_mod.settings, "images_dir", tmp_path / "images")
    monkeypatch.setattr(cfg_mod.settings, "thumbnails_dir", tmp_path / "images" / "thumbs")
    monkeypatch.setattr(cfg_mod.settings, "models_json", tmp_path / "models.json")

    # Use a file-based SQLite in tmp_path — avoids multi-connection in-memory isolation issues
    db_path = tmp_path / "test.db"
    test_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    test_session_factory = async_sessionmaker(test_engine, expire_on_commit=False)

    monkeypatch.setattr("backend.gallery.db.engine", test_engine)
    monkeypatch.setattr("backend.gallery.db.AsyncSessionLocal", test_session_factory)

    # Patch pipeline loader so no real model is loaded
    monkeypatch.setattr(
        "backend.model_registry.loader.PipelineLoader.load",
        lambda self, entry: mock_pipeline,
    )

    # Build a fresh FastAPI app (bypasses module-level singletons)
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from backend.queue.manager import QueueManager
    from backend.model_registry.registry import ModelRegistry
    from backend.model_registry.loader import PipelineLoader
    from backend.pipeline.preview import PreviewBroadcaster
    from backend.pipeline.worker import PipelineWorker
    from backend.api import queue, images, boards, models, websocket
    from backend.gallery.db import init_db
    import asyncio

    _queue = QueueManager()
    _registry = ModelRegistry(tmp_path / "models.json")
    _loader = PipelineLoader()
    _broadcaster = PreviewBroadcaster()
    _worker = PipelineWorker(
        queue=_queue,
        registry=_registry,
        loader=_loader,
        broadcaster=_broadcaster,
        session_factory=test_session_factory,
        default_board_id=None,
    )

    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        (tmp_path / "images").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / "thumbs").mkdir(parents=True, exist_ok=True)
        # init_db is already done above, but call it to stay consistent
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # Create default gallery
        from backend.gallery.service import GalleryService
        async with test_session_factory() as session:
            svc = GalleryService(session)
            galleries = await svc.list_galleries()
            if galleries:
                _worker._default_board_id = galleries[0].board_id
            else:
                g = await svc.create_gallery("Default")
                await session.commit()
                _worker._default_board_id = g.board_id
        task = asyncio.create_task(_worker.run())
        yield
        await _worker.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    test_app = FastAPI(title="diffusers-ui-test", lifespan=test_lifespan)
    test_app.state.queue = _queue
    test_app.state.registry = _registry
    test_app.state.broadcaster = _broadcaster

    test_app.include_router(queue.router)
    test_app.include_router(images.router)
    test_app.include_router(boards.router)
    test_app.include_router(models.router)
    test_app.include_router(websocket.router)

    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as c:
        yield c

    await test_engine.dispose()


def make_enqueue_body(**overrides) -> dict:
    """Build a valid enqueue_batch request body."""
    defaults = dict(
        prepend=False,
        batch=dict(
            prompt="a beautiful landscape",
            negative_prompt="",
            seed=42,
            width=512,
            height=512,
            steps=20,
            cfg=7.5,
            cfg_rescale_multiplier=0.0,
            sampler="euler_a",
            model_key="test_model",
        ),
    )
    defaults.update(overrides)
    return defaults




