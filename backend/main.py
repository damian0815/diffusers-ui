from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.gallery.db import init_db, AsyncSessionLocal
from backend.queue.manager import QueueManager
from backend.model_registry.registry import ModelRegistry
from backend.model_registry.loader import PipelineLoader
from backend.pipeline.preview import PreviewBroadcaster
from backend.pipeline.worker import PipelineWorker
from backend.api import queue, images, boards, models, websocket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application-level singletons (shared across requests via app.state)
# ---------------------------------------------------------------------------
queue_manager = QueueManager()
model_registry = ModelRegistry(settings.models_json)
pipeline_loader = PipelineLoader()
broadcaster = PreviewBroadcaster()
worker = PipelineWorker(
    queue=queue_manager,
    registry=model_registry,
    loader=pipeline_loader,
    broadcaster=broadcaster,
    session_factory=AsyncSessionLocal,
    default_board_id=None,  # set during lifespan startup
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown sequence."""
    # Ensure required directories exist
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)

    # Set up database
    await init_db()

    # Create default gallery if none exist
    default_board_id = await _ensure_default_gallery()
    worker._default_board_id = default_board_id

    logger.info("Detected device: %s", settings.device)
    logger.info("Default gallery board_id: %s", default_board_id)

    # Start background worker
    task = asyncio.create_task(worker.run())

    yield

    # Graceful shutdown
    await worker.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _ensure_default_gallery() -> str:
    """Create the 'Default' gallery if it does not exist. Return its board_id."""
    from backend.gallery.service import GalleryService
    async with AsyncSessionLocal() as session:
        service = GalleryService(session)
        galleries = await service.list_galleries()
        if galleries:
            await session.commit()
            return galleries[0].board_id
        gallery = await service.create_gallery("Default")
        await session.commit()
        return gallery.board_id


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------
app = FastAPI(title="diffusers-ui", lifespan=lifespan)

app.state.queue = queue_manager
app.state.registry = model_registry
app.state.broadcaster = broadcaster

app.include_router(queue.router)
app.include_router(images.router)
app.include_router(boards.router)
app.include_router(models.router)
app.include_router(websocket.router)

# Serve the frontend as static files (must be last — catch-all)
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

