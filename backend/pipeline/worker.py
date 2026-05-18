from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image as PILImage

from backend.config import settings
from backend.domain import QueueItem, ItemStatus
from backend.pipeline.batcher import group_into_batches
from backend.pipeline.preview import latents_to_preview_image, PreviewBroadcaster
from backend.pipeline.compel_encoder import build_prompt_embeds

if TYPE_CHECKING:
    from backend.queue.manager import QueueManager
    from backend.model_registry.loader import PipelineLoader
    from backend.model_registry.registry import ModelRegistry

from backend.pipeline.flow_match_scheduler import SDPipelineInferenceFlowMatchEulerDiscreteScheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag: use the scheduler's predicted-original-sample for previews
# instead of the current (noisy) latent.
#
# When True the step callback requests "pred_original_sample" from the
# scheduler and, if it is available, passes it to latents_to_preview_image.
# This gives a much cleaner preview from step ~10 onward, but not every
# scheduler exposes pred_original_sample — those fall back gracefully to the
# noisy latent.
#
# See PRED_ORIGINAL_SAMPLE_howto_enable.md for how to enable this.
# ---------------------------------------------------------------------------
USE_PRED_ORIGINAL_SAMPLE: bool = True  # ← set to True to enable

# Map domain sampler names → diffusers scheduler class names
SAMPLER_MAP: dict[str, str] = {
    "dpmpp_sde_k": "DPMSolverSDEScheduler",
    "dpmpp_sde": "DPMSolverSDEScheduler",
    "dpmpp_2m_sde_k": "DPMSolverMultistepScheduler",
    "dpmpp_2m_sde": "DPMSolverMultistepScheduler",
    "dpmpp_2m": "DPMSolverMultistepScheduler",
    "dpmpp_3m": "DPMSolverMultistepScheduler",
    "euler": "EulerDiscreteScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "euler_fm_dynshift": "SDPipelineInferenceFlowMatchEulerDiscreteScheduler",
    "ddim": "DDIMScheduler",
    "lms": "LMSDiscreteScheduler",
    "kdpm_2_a": "KDPM2AncestralDiscreteScheduler",
}

# Samplers that use Karras sigma sigmas
KARRAS_SAMPLERS = {"dpmpp_sde_k", "dpmpp_2m_sde_k"}


def _is_oom(exc: Exception) -> bool:
    """Return True if *exc* looks like a GPU/CPU out-of-memory error."""
    # torch.cuda.OutOfMemoryError is a RuntimeError subclass (PyTorch ≥ 2.0)
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "outofmemory" in msg


class PipelineWorker:
    """
    Runs as a single asyncio background task.
    Drains the queue in batches and calls the diffusers pipeline.
    Saves output images to disk and records them in the gallery database.
    """

    def __init__(
        self,
        queue: QueueManager,
        registry: ModelRegistry,
        loader: PipelineLoader,
        broadcaster: PreviewBroadcaster,
        session_factory=None,
        default_board_id: str | None = None,
    ):
        self._queue = queue
        self._registry = registry
        self._loader = loader
        self._broadcaster = broadcaster
        self._session_factory = session_factory
        self._default_board_id = default_board_id
        self._running = False
        # _idle is SET when no generation is in flight; CLEARED while running one.
        self._idle = asyncio.Event()
        self._idle.set()
        # Maximum items per pipeline call; 0 = unlimited.
        # Automatically halved on CUDA/MPS out-of-memory to avoid repeated OOM.
        self._max_batch_size: int = 0

    async def run(self) -> None:
        """Main loop — drains queue continuously until stopped."""
        self._running = True
        while self._running:
            try:
                items = await self._queue.next_pending_batch_window(settings.batch_window_s)
            except asyncio.CancelledError:
                break
            if not items:
                continue
            for batch in group_into_batches(items, max_size=self._max_batch_size):
                if not self._running:
                    break
                self._idle.clear()
                try:
                    await self._run_batch(batch)
                except Exception:
                    logger.exception("Unhandled error in batch, continuing.")
                finally:
                    self._idle.set()

    async def stop(self) -> None:
        """Signal the worker to stop and unblock any queue wait."""
        self._running = False
        self._queue.abort_wait()

    async def wait_idle(self) -> None:
        """Wait until no generation is actively running (safe to exit after this)."""
        await self._idle.wait()

    async def _run_batch(self, batch: list[QueueItem]) -> None:
        """Run a single batch: load model, generate, save, record."""
        entry = self._registry.get(batch[0].params.model_key)
        if entry is None:
            error_msg = f"Model key {batch[0].params.model_key!r} not found in registry"
            for item in batch:
                await self._queue.mark_failed(item, error_msg)
            logger.error(error_msg)
            return

        try:
            pipeline = await asyncio.to_thread(self._loader.load, entry)
        except Exception as exc:
            for item in batch:
                await self._queue.mark_failed(item, str(exc))
            logger.error("Pipeline load failed: %s", exc)
            return

        # Swap scheduler based on sampler
        await asyncio.to_thread(self._apply_scheduler, pipeline, batch[0].params.sampler)

        for item in batch:
            await self._queue.mark_processing(item)

        params = batch[0].params  # all share critical params
        prompts = [item.params.prompt for item in batch]
        neg_prompts = [item.params.negative_prompt for item in batch]
        seeds = [item.params.seed for item in batch]
        is_sdxl: bool = entry.base == "sdxl"

        # Encode prompts with Compel (supports weighted syntax and long prompts).
        # Runs in a thread because text encoding calls the GPU-bound text encoder.
        prompt_kwargs = await asyncio.to_thread(
            build_prompt_embeds, pipeline, prompts, neg_prompts, is_sdxl=is_sdxl
        )

        step_counter: dict[str, int] = {"n": 0}
        event_loop = asyncio.get_event_loop()

        def on_step(pipe, step: int, timestep, callback_kwargs: dict) -> dict:
            step_counter["n"] += 1
            if step_counter["n"] % settings.preview_every_n_steps == 0:
                # ------------------------------------------------------------------
                # USE_PRED_ORIGINAL_SAMPLE block (disabled by default).
                # When enabled, prefer the scheduler's denoised estimate over the
                # raw noisy latent — produces a cleaner preview from ~step 10 on.
                # Sources checked in order:
                #   1. pipe.scheduler._last_pred_original_sample — set by our
                #      SDPipelineInferenceFlowMatchEulerDiscreteScheduler.step()
                #      override (and any future scheduler overrides).
                #      NOTE: callback_on_step_end_tensor_inputs cannot include
                #      "pred_original_sample" because diffusers validates that list
                #      against a fixed pipeline-defined whitelist; the scheduler
                #      attribute is the only reliable cross-scheduler path.
                #   2. callback_kwargs["latents"] — noisy-latent fallback for
                #      schedulers that set no attribute.
                # See PRED_ORIGINAL_SAMPLE_howto_enable.md to enable.
                # ------------------------------------------------------------------
                if USE_PRED_ORIGINAL_SAMPLE:
                    sample = getattr(pipe.scheduler, "_last_pred_original_sample", None)
                    if sample is None:
                        # Scheduler doesn't expose it — fall back gracefully.
                        sample = callback_kwargs.get("latents")
                else:
                    sample = callback_kwargs.get("latents")
                # ------------------------------------------------------------------

                if sample is not None:
                    for i, item in enumerate(batch):
                        try:
                            preview = latents_to_preview_image(
                                sample[i: i + 1], is_sdxl=is_sdxl
                            )
                            asyncio.run_coroutine_threadsafe(
                                self._broadcaster.broadcast(
                                    item.item_id, step, params.steps, preview
                                ),
                                event_loop,
                            )
                        except Exception:
                            pass  # preview failure is non-fatal
            return callback_kwargs

        try:
            import torch
            generators = [
                torch.Generator(device=settings.device).manual_seed(s)
                for s in seeds
            ]
            if isinstance(pipeline.scheduler,
                          SDPipelineInferenceFlowMatchEulerDiscreteScheduler) and pipeline.scheduler.config.use_dynamic_shifting:
                image_pixel_count = params.width * params.height
                # for linear shift, we go from 1 to 3 over 1 megapixel
                assert pipeline.scheduler.config.time_shift_type == 'linear'
                shift = 1.0 + 0.5 * (image_pixel_count / 1024 ** 2)
                pipeline.scheduler.set_shift(shift)
                # pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, shift=shift)
                # print("updated scheduler with shift", shift)

            result = await asyncio.to_thread(
                pipeline,
                **prompt_kwargs,
                width=params.width,
                height=params.height,
                num_inference_steps=params.steps,
                guidance_scale=params.cfg,
                generator=generators,
                callback_on_step_end=on_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
            for item, pil_image in zip(batch, result.images):
                image_name = await asyncio.to_thread(_save_image, pil_image)
                await self._queue.mark_completed(item, image_name)
                await self._record_in_gallery(item, image_name, entry.name)

        except Exception as exc:
            if _is_oom(exc):
                await self._handle_oom(batch, exc)
            else:
                logger.exception("Pipeline inference failed")
                for item in batch:
                    if item.status == ItemStatus.PROCESSING:
                        await self._queue.mark_failed(item, str(exc))

    async def _handle_oom(self, batch: list[QueueItem], exc: Exception) -> None:
        """
        Respond to an out-of-memory error during inference.

        Strategy:
        - Free the VRAM cache immediately.
        - If the batch has more than one item, halve ``_max_batch_size`` and
          re-enqueue all items at the front of the queue so they are retried
          at a smaller batch size on the next pass.
        - If the batch is already a single item, it can't be split further;
          mark it as failed with a clear OOM message.
        """
        import torch
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if len(batch) > 1:
            new_max = max(1, len(batch) // 2)
            # Only ever reduce, never increase.
            if self._max_batch_size == 0 or new_max < self._max_batch_size:
                self._max_batch_size = new_max
            logger.warning(
                "OOM with batch size %d — reducing max batch size to %d and requeuing.",
                len(batch), self._max_batch_size,
            )
            await self._queue.requeue_items(batch, prepend=True)
        else:
            logger.error("OOM on single-item batch — failing item %d.", batch[0].item_id)
            await self._queue.mark_failed(
                batch[0],
                "Out of memory — try reducing image size or number of steps.",
            )

    def _apply_scheduler(self, pipeline, sampler: str) -> None:
        """Swap the pipeline scheduler in-place based on the sampler name."""
        scheduler_class_name = SAMPLER_MAP.get(sampler)
        if scheduler_class_name is None:
            return
        try:
            config = dict(pipeline.scheduler.config)
            if sampler == 'euler_fm_dynshift':
                pipeline.scheduler = SDPipelineInferenceFlowMatchEulerDiscreteScheduler.from_config(
                    config, shift=1, use_dynamic_shifting=True, time_shift_type='linear'
                )
            else:
                import diffusers
                scheduler_cls = getattr(diffusers, scheduler_class_name, None)
                if scheduler_cls is None:
                    return
                use_karras = sampler in KARRAS_SAMPLERS
                if use_karras:
                    config["use_karras_sigmas"] = True
                pipeline.scheduler = scheduler_cls.from_config(config)
        except Exception:
            logger.warning("Could not swap scheduler for sampler %r", sampler)

    async def _record_in_gallery(
        self, item: QueueItem, image_name: str, model_name: str
    ) -> None:
        """Record a completed image in the gallery database."""
        if self._session_factory is None or self._default_board_id is None:
            return
        try:
            from backend.gallery.service import GalleryService
            async with self._session_factory() as session:
                service = GalleryService(session)
                await service.record_image(
                    item, image_name, self._default_board_id, model_name
                )
                await session.commit()
        except Exception:
            logger.exception("Failed to record image %r in gallery", image_name)


def _save_image(pil_image: PILImage.Image) -> str:
    """Save image + thumbnail; return the image filename."""
    name = f"{uuid.uuid4().hex}.png"

    settings.images_dir.mkdir(parents=True, exist_ok=True)
    pil_image.save(settings.images_dir / name)

    thumb = pil_image.copy()
    thumb.thumbnail(settings.thumbnail_size)
    settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    thumb.save(settings.thumbnails_dir / name)

    return name

