"""Unit tests for PipelineWorker — submission serial preservation."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from backend.config import settings
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


# ---------------------------------------------------------------------------
# VAE slicing (queue-level OOM fallback)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_force_enable_vae_slicing_enables_and_persists(worker_ctx):
    _, _, _, worker = worker_ctx
    worker.save_batch_state = MagicMock()
    worker._loader.enable_vae_slicing_on_loaded = MagicMock()

    worker._force_enable_vae_slicing()

    assert worker.vae_slicing is True
    worker.save_batch_state.assert_called_once()
    worker._loader.enable_vae_slicing_on_loaded.assert_called_once()


@pytest.mark.asyncio
async def test_force_enable_vae_slicing_is_idempotent(worker_ctx):
    _, _, _, worker = worker_ctx
    worker.save_batch_state = MagicMock()
    worker._loader.enable_vae_slicing_on_loaded = MagicMock()

    worker._force_enable_vae_slicing()
    worker._force_enable_vae_slicing()

    assert worker._loader.enable_vae_slicing_on_loaded.call_count == 1


@pytest.mark.asyncio
async def test_oom_at_batch_over_one_force_enables_slicing(worker_ctx):
    _, _, _, worker = worker_ctx
    worker.save_batch_state = MagicMock()
    worker._loader.enable_vae_slicing_on_loaded = MagicMock()
    worker._run_batch = AsyncMock(return_value=False)
    worker._queue.requeue_items = AsyncMock()

    batch = [QueueItem(params=make_params()), QueueItem(params=make_params())]
    await worker._handle_oom(batch, RuntimeError("CUDA out of memory"))

    assert worker.vae_slicing is True
    worker._loader.enable_vae_slicing_on_loaded.assert_called_once()
    worker._queue.requeue_items.assert_awaited_once()
    worker._run_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_oom_at_batch_size_one_does_not_force_slicing(worker_ctx):
    _, _, _, worker = worker_ctx
    worker.save_batch_state = MagicMock()
    worker._loader.enable_vae_slicing_on_loaded = MagicMock()
    worker._queue.mark_failed = AsyncMock()

    batch = [QueueItem(params=make_params())]
    await worker._handle_oom(batch, RuntimeError("CUDA out of memory"))

    assert worker.vae_slicing is False
    worker._loader.enable_vae_slicing_on_loaded.assert_not_called()


def test_mem_advisor_vae_slicing_persists(tmp_path):
    from backend.pipeline.mem_stats import MemoryAdvisor
    advisor = MemoryAdvisor(device="cpu")
    path = tmp_path / "batch_state.json"
    advisor.save_state(path)
    advisor.vae_slicing = True
    advisor.save_state(path)
    advisor.vae_slicing = False
    advisor.load_state(path)
    assert advisor.vae_slicing is True


def test_mem_advisor_loads_without_vae_slicing_key(tmp_path):
    from backend.pipeline.mem_stats import MemoryAdvisor
    advisor = MemoryAdvisor(device="cpu")
    path = tmp_path / "batch_state.json"
    path.write_text('{"max_batch_size_override": 4, "probe_batch_size": 2}')
    advisor.load_state(path)
    assert advisor.vae_slicing is False


# ---------------------------------------------------------------------------
# LoRA loading / activation
# ---------------------------------------------------------------------------

class FakePipeline:
    """Stand-in for a diffusers pipeline; records LoRA method calls."""

    def __init__(self):
        self.load_lora_weights = MagicMock()
        self.unload_lora_weights = MagicMock()
        self.unload_lora = MagicMock()
        self.enable_lora = MagicMock()
        self.set_adapters = MagicMock()
        self.disable_lora = MagicMock()


def _item_with_lora(filename, strength):
    return QueueItem(params=make_params(lora_keys=[filename], lora_strengths=[strength]))


@pytest.mark.asyncio
async def test_load_loras_loads_and_activates_new_lora(worker_ctx):
    _, _, _, worker = worker_ctx
    pipeline = FakePipeline()
    filename = "my_lora.safetensors"

    await worker._load_loras(pipeline, [_item_with_lora(filename, 0.7)])

    pipeline.load_lora_weights.assert_called_once_with(
        settings.loras_dir, weight_name=filename, adapter_name="lora_adapter_0"
    )
    pipeline.enable_lora.assert_called_once()
    pipeline.set_adapters.assert_called_once_with(["lora_adapter_0"], adapter_weights=[0.7])
    assert worker._loaded_loras == {filename}
    pipeline.disable_lora.assert_not_called()


@pytest.mark.asyncio
async def test_load_loras_same_set_only_updates_weights(worker_ctx):
    _, _, _, worker = worker_ctx
    pipeline = FakePipeline()
    filename = "my_lora.safetensors"

    await worker._load_loras(pipeline, [_item_with_lora(filename, 0.5)])
    await worker._load_loras(pipeline, [_item_with_lora(filename, 0.9)])

    pipeline.load_lora_weights.assert_called_once()
    pipeline.set_adapters.assert_called_with(["lora_adapter_0"], adapter_weights=[0.9])
    pipeline.enable_lora.assert_called()
    pipeline.disable_lora.assert_not_called()


@pytest.mark.asyncio
async def test_load_loras_unloads_with_unload_lora_weights(worker_ctx):
    _, _, _, worker = worker_ctx
    pipeline = FakePipeline()
    filename = "my_lora.safetensors"

    await worker._load_loras(pipeline, [_item_with_lora(filename, 0.5)])
    await worker._load_loras(pipeline, [QueueItem(params=make_params())])

    pipeline.unload_lora_weights.assert_called_once()
    pipeline.unload_lora.assert_not_called()
    assert worker._loaded_loras == set()
    pipeline.disable_lora.assert_not_called()


@pytest.mark.asyncio
async def test_load_loras_skips_failed_load_but_keeps_others(worker_ctx):
    _, _, _, worker = worker_ctx
    pipeline = FakePipeline()
    good, bad = "good.safetensors", "bad.safetensors"
    pipeline.load_lora_weights.side_effect = (
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad file"))
        if k.get("weight_name") == bad
        else None
    )
    item = QueueItem(params=make_params(lora_keys=[good, bad], lora_strengths=[1.0, 1.0]))

    await worker._load_loras(pipeline, [item])

    assert worker._loaded_loras == {good}
    pipeline.set_adapters.assert_called_once_with(["lora_adapter_1"], adapter_weights=[1.0])
