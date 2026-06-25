from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.domain import _init_counter
from backend.gallery.db import init_db, AsyncSessionLocal
from backend.gallery.models import Gallery
from backend.queue.manager import QueueManager
from backend.queue.store import QueueStore
from backend.model_registry.registry import ModelRegistry
from backend.model_registry.loader import PipelineLoader
from backend.pipeline.preview import PreviewBroadcaster
from backend.pipeline.worker import PipelineWorker
from backend.api import queue, images, boards, models, websocket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application-level singletons (shared across requests via app.state)
# ---------------------------------------------------------------------------
queue_store   = QueueStore(session_factory=AsyncSessionLocal)
queue_manager = QueueManager(store=queue_store)
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

    # Set up database (creates tables if they don't exist)
    await init_db()

    await _cleanup_broken_images()

    # Create default gallery if none exist
    default_board_id = await _ensure_default_gallery()
    worker._default_board_id = default_board_id

    # Ensure the img2img source-image board exists (created lazily by
    # record_image too, but eager creation documents the intent).
    from backend.gallery.service import GalleryService, IMG2IMG_BOARD_NAME
    img2img_board_id = await GalleryService.ensure_img2img_board()
    logger.info("Img2img source board (%r) id: %s", IMG2IMG_BOARD_NAME, img2img_board_id)

    logger.info("Detected device: %s", settings.device)
    logger.info("Default gallery board_id: %s", default_board_id)

    # Restore persisted queue state ------------------------------------------
    await _restore_queue()

    # Start background worker
    task = asyncio.create_task(worker.run())

    yield

    # ── Graceful shutdown ──────────────────────────────────────────────────
    logger.info("Shutdown requested — waiting for current generation to finish…")

    # 1. Tell the worker to stop accepting new items and unblock its queue wait.
    await worker.stop()

    # 2. Wait for any in-flight generation to complete (up to 10 minutes).
    try:
        await asyncio.wait_for(worker.wait_idle(), timeout=600.0)
        logger.info("Current generation finished cleanly.")
    except asyncio.TimeoutError:
        logger.warning("Shutdown timeout exceeded — forcing exit.")

    # 3. Cancel the now-idle worker task.
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    logger.info("Shutdown complete.")


async def _restore_queue() -> None:
    """Load persisted queue items from the DB and repopulate the in-memory queue."""
    try:
        # Any item left in 'processing' state means the server was killed mid-run;
        # reset it to 'pending' so it will be re-generated on this run.
        await queue_store.reset_interrupted()

        items = await queue_store.load_all()
        if not items:
            return

        pending_count = 0
        for item in items:
            queue_manager._all_items[item.item_id] = item
            if item.status.value == "pending":
                queue_manager._queue.append(item)
                pending_count += 1

        # Prevent future IDs from colliding with restored IDs.
        max_id = max(i.item_id for i in items)
        _init_counter(max_id)

        # Wake the worker if there are pending items to process.
        if pending_count:
            queue_manager._new_item_event.set()

        logger.info(
            "Restored %d queue items from DB (%d pending).",
            len(items),
            pending_count,
        )
    except Exception:
        logger.exception("Failed to restore queue from DB — starting with empty queue.")


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


async def _cleanup_broken_images():
    """ Remove all files that are not known to any galleries """
    logger.warning("Cleaning up broken images...")
    from sqlalchemy import select
    from backend.gallery.models import ImageRecord
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ImageRecord.image_name))
        all_known_images = {row[0] for row in result.all()}

        num_removed = 0
        for image_path in list(settings.images_dir.glob('*.png')):
            if image_path.name not in all_known_images:
                logger.warning(f"Removing broken image file: {image_path}")
                try:
                    image_path.unlink()
                    thumbnail_path = settings.thumbnails_dir / image_path.name
                    thumbnail_path.unlink()
                    num_removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove broken image file {image_path}: {e}")
        logger.warning(f"Removed {num_removed} broken images out of {len(all_known_images)} total.")

# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------
app = FastAPI(title="diffusers-ui", lifespan=lifespan)

app.state.queue = queue_manager
app.state.registry = model_registry
app.state.broadcaster = broadcaster
app.state.loader = pipeline_loader
app.state.worker = worker

app.include_router(queue.router)
app.include_router(images.router)
app.include_router(boards.router)
app.include_router(models.router)
app.include_router(websocket.router)

# Serve the frontend as static files (must be last — catch-all)
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
