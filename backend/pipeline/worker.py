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

if TYPE_CHECKING:
    from backend.queue.manager import QueueManager
    from backend.model_registry.loader import PipelineLoader
    from backend.model_registry.registry import ModelRegistry

from backend.pipeline.flow_match_scheduler import SDPipelineInferenceFlowMatchEulerDiscreteScheduler

logger = logging.getLogger(__name__)

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
            for batch in group_into_batches(items):
                if not self._running:
                    break
                try:
                    await self._run_batch(batch)
                except Exception:
                    logger.exception("Unhandled error in batch, continuing.")

    async def stop(self) -> None:
        self._running = False

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

        step_counter: dict[str, int] = {"n": 0}
        event_loop = asyncio.get_event_loop()

        def on_step(pipe, step: int, timestep, callback_kwargs: dict) -> dict:
            step_counter["n"] += 1
            if step_counter["n"] % settings.preview_every_n_steps == 0:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    for i, item in enumerate(batch):
                        try:
                            preview = latents_to_preview_image(latents[i: i + 1])
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
                prompt=prompts,
                negative_prompt=neg_prompts,
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
            logger.exception("Pipeline inference failed")
            for item in batch:
                if item.status == ItemStatus.PROCESSING:
                    await self._queue.mark_failed(item, str(exc))

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

