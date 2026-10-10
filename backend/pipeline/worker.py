from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
import types
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image as PILImage
import diffusers
from diffusers import SanaPAGPipeline

from backend.config import settings
from backend.domain import GenerationCancelled, QueueItem, ItemStatus
from backend.pipeline.batcher import get_critical_params
from backend.pipeline.preview import latents_to_preview_image, PreviewBroadcaster
from backend.pipeline.compel_encoder import build_prompt_embeds, build_sana_prompt_embeds
from backend.pipeline.mem_stats import MemoryAdvisor
from backend.prompt_utils import EMBEDDING_RE, expand_prompt

_BATCH_STATE_FILENAME = "batch_state.json"

_WINDOW_SIZE = 20  # generations used for images/sec calculation


class _GenerationTimingTracker:
    """Rolling-window tracker for generation speed (images/sec)."""

    def __init__(self) -> None:
        self._samples: collections.deque[tuple[float, int]] = collections.deque(maxlen=_WINDOW_SIZE)

    def record(self, elapsed_seconds: float, batch_size: int) -> None:
        if elapsed_seconds > 0 and batch_size > 0:
            self._samples.append((elapsed_seconds, batch_size))

    @property
    def images_per_sec(self) -> float:
        if not self._samples:
            return 0.0
        total_images = sum(bs for _, bs in self._samples)
        total_seconds = sum(t for t, _ in self._samples)
        return total_images / total_seconds if total_seconds > 0 else 0.0

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
SAMPLER_MAP: dict[str, tuple[str, dict]] = {
    "dpmpp": ("DPMSolverMultistepScheduler", dict(
        algorithm_type="dpmsolver++", use_karras_sigmas=False, solver_order=1
    )),
    "dpmpp_sde_k": ("DPMSolverSDEScheduler", dict(use_karras_sigmas=True)),
    "dpmpp_sde": ("DPMSolverSDEScheduler", dict(use_karras_sigmas=False)),
    "dpmpp_2m_sde_k": ("DPMSolverMultistepScheduler", dict(
        algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, solver_order=2
    )),
    "dpmpp_2m_sde": ("DPMSolverMultistepScheduler", dict(
        algorithm_type="sde-dpmsolver++", use_karras_sigmas=False, solver_order=2
    )),
    "dpmpp_2m": ("DPMSolverMultistepScheduler", dict(
        algorithm_type="dpmsolver++", use_karras_sigmas=False, solver_order=2
    )),
    "dpmpp_3m": ("DPMSolverMultistepScheduler", dict(
        algorithm_type="dpmsolver++", use_karras_sigmas=False, solver_order=3
    )),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "euler_flowmatch_dynshift": ("SDPipelineInferenceFlowMatchEulerDiscreteScheduler", dict(shift=1, use_dynamic_shifting=True, time_shift_type='linear')),
    "euler_flowmatch_shift1": ("SDPipelineInferenceFlowMatchEulerDiscreteScheduler", dict(shift=1, use_dynamic_shifting=False, time_shift_type='linear')),
    "ddim": ("DDIMScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "kdpm_2_a": ("KDPM2AncestralDiscreteScheduler", {}),
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
        embedding_service=None,
    ):
        self._queue = queue
        self._registry = registry
        self._loader = loader
        self._broadcaster = broadcaster
        self._session_factory = session_factory
        self._default_board_id = default_board_id
        self._embedding_service = embedding_service
        self._running = False
        # _idle is SET when no generation is in flight; CLEARED while running one.
        self._idle = asyncio.Event()
        self._idle.set()
        # _paused is SET when the worker may pick up new work; CLEARED on pause.
        # The current batch always finishes before pausing takes effect.
        self._paused = asyncio.Event()
        self._paused.set()
        # Thread-safe event signalled by cancel_generation() and checked in the
        # pipeline step callback (which runs in a thread-pool thread).
        self._cancel_event = threading.Event()
        # Adaptive batch-size advisor: starts at probe=1, ramps up by +1 on each
        # successful run, regresses on OOM, and fits a power-law memory model.
        self._mem_advisor = MemoryAdvisor(device=settings.device)
        # Rolling-window timing tracker for ETA calculation.
        self._timing_tracker = _GenerationTimingTracker()
        # Key of the model currently loaded in the PipelineLoader. Used to
        # prefer same-model work when choosing the next batch, minimising
        # model swaps. None until the first batch loads a model.
        self._loaded_model_key: str | None = None
        # Track currently loaded LoRAs to avoid redundant load/unload cycles.
        self._loaded_loras: set[str] = set()

    async def run(self) -> None:
        """
        Main loop — scan the full queue depth to find compatible items and
        batch them together, then run inference.
        """
        self._running = True
        while self._running:
            # Pause before dequeuing the next batch (current batch always
            # completes first).
            await self._paused.wait()

            try:
                # Get exactly one item (blocks until available). When a model
                # is already loaded, prefer a pending item that uses it so we
                # minimise model-swap time.
                items = await self._queue.next_pending_batch_window(
                    0, limit=1, preferred_model_key=self._loaded_model_key
                )
            except asyncio.CancelledError:
                break
            if not items:
                continue

            pivot = items[0]

            crit = get_critical_params(pivot.params)
            max_bs = self._mem_advisor.get_max_batch_size(crit.width * crit.height)
            batch = [pivot]
            if max_bs > 1:
                if pivot.priority:
                    rest = await self._queue.dequeue_matching(
                        lambda item, _crit=crit: item.priority and get_critical_params(item.params) == _crit,
                        max_bs - 1,
                    )
                else:
                    rest = await self._queue.dequeue_matching(
                        lambda item, _crit=crit: not item.priority and get_critical_params(item.params) == _crit,
                        max_bs - 1,
                    )
                batch.extend(rest)

            if not self._running:
                break
            self._idle.clear()
            try:
                _ = await self._run_batch(batch)
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

    @property
    def is_idle(self) -> bool:
        """True when no generation is currently in flight."""
        return self._idle.is_set()

    @property
    def is_paused(self) -> bool:
        """True when the worker is paused (won't start new batches)."""
        return not self._paused.is_set()

    def pause(self) -> None:
        """Pause the worker after the current batch finishes."""
        self._paused.clear()
        logger.info("Worker paused — will not pick up new batches.")

    def resume(self) -> None:
        """Resume the worker so it picks up queued items again."""
        self._paused.set()
        logger.info("Worker resumed.")

    @property
    def max_batch_size(self) -> int:
        """Current user-set cap on batch size (0 = auto)."""
        return self._mem_advisor.max_batch_size_override

    @property
    def vae_slicing(self) -> bool:
        """Whether VAE slicing is forced on (batch-dimension decode split)."""
        return self._mem_advisor.vae_slicing

    @property
    def images_per_sec(self) -> float:
        """Rolling-window average images/sec over recent generations."""
        return self._timing_tracker.images_per_sec

    @property
    def _batch_state_path(self) -> Path:
        return settings.images_dir / _BATCH_STATE_FILENAME

    def save_batch_state(self) -> None:
        """Persist batch-size settings so they survive restarts."""
        try:
            self._mem_advisor.save_state(self._batch_state_path)
        except Exception:
            logger.warning("Failed to persist batch-size state", exc_info=True)

    def load_batch_state(self) -> None:
        """Load persisted batch-size settings (if available)."""
        try:
            self._mem_advisor.load_state(self._batch_state_path)
        except Exception:
            logger.warning("Failed to load batch-size state", exc_info=True)

    def set_max_batch_size(self, n: int) -> None:
        """Set a user cap on batch size (0 = auto, let the advisor decide).

        If the requested value is larger than the current probe the probe is
        bumped immediately so the new value takes effect on the next batch.
        If the requested value is smaller than or equal to the probe, the
        override is kept so the UI accurately reflects the chosen cap while
        the adaptive probe restarts from that value.
        """
        n = max(0, n)
        advisor = self._mem_advisor
        if n == 0:
            # Auto — clear any user cap, keep current probe.
            advisor.max_batch_size_override = 0
            logger.info("Max batch size set to 0 (auto, probe=%d)", advisor._probe_batch_size)
        elif n > advisor._probe_batch_size:
            # User wants larger batches than we've discovered.
            advisor.max_batch_size_override = n
            advisor._probe_batch_size = n
            logger.info("Max batch size set to %d (larger than probe; probe bumped)", n)
        else:
            # User wants smaller batches — start discovery from this point.
            advisor.max_batch_size_override = n
            advisor._probe_batch_size = n
            advisor._max_batch_size_is_known = False
            advisor._gen_count_at_current_batch_size = 0
            logger.info("Max batch size set to %d (reverting to discovery mode)", n)
        self.save_batch_state()

    def cancel_generation(self) -> None:
        """Request that the current in-flight generation be aborted.

        Thread-safe — sets a ``threading.Event`` that the pipeline step callback
        checks on every denoising step.  Returns immediately; the worker will
        cancel the current batch asynchronously.
        """
        self._cancel_event.set()

    def set_vae_slicing(self, enabled: bool) -> None:
        """Set the queue-level VAE slicing flag and persist it.

        Enabling also applies immediately to the currently loaded pipeline so
        the very next generation benefits. Disabling takes effect on the next
        model load (VAE slicing stays active on the loaded pipeline until then).
        """
        enabled = bool(enabled)
        self._mem_advisor.vae_slicing = enabled
        self.save_batch_state()
        if enabled:
            try:
                self._loader.enable_vae_slicing_on_loaded()
            except Exception:
                logger.exception("Failed to enable VAE slicing on the loaded pipeline")
        logger.info("VAE slicing %s", "enabled" if enabled else "disabled")

    def _force_enable_vae_slicing(self) -> None:
        """OOM fallback: turn on VAE slicing (persisted) unless already on."""
        if self.vae_slicing:
            return
        logger.warning("Force-enabling VAE slicing as an OOM fallback.")
        self.set_vae_slicing(True)

    async def _load_loras(self, pipeline, batch: list[QueueItem]) -> None:
        """Load and activate LoRAs for the current batch.

        Compares the batch's LoRA set against the currently loaded set and
        only loads/unloads when the set changes. After loading, calls
        set_adapters with per-item weights.
        """
        # Collect unique LoRA filenames from the batch
        batch_loras: dict[str, float] = {}
        for item in batch:
            for key, strength in zip(item.params.lora_keys, item.params.lora_strengths):
                if key not in batch_loras:
                    batch_loras[key] = strength

        batch_lora_set = set(batch_loras.keys())
        # Assign short, stable adapter names. Sorted so the same set always maps
        # the same file to the same name across batch boundaries.
        adapter_name_for = {
            filename: f"lora_adapter_{i}" for i, filename in enumerate(sorted(batch_lora_set))
        }

        # Skip if LoRAs haven't changed
        if batch_lora_set == self._loaded_loras:
            # Same set — just update weights (they may differ per batch)
            if batch_lora_set:
                adapter_names = [adapter_name_for[f] for f in batch_lora_set]
                weights = [batch_loras[f] for f in batch_lora_set]
                await asyncio.to_thread(pipeline.enable_lora)
                await asyncio.to_thread(
                    pipeline.set_adapters, adapter_names, adapter_weights=weights
                )
            return

        # Unload previous LoRAs if any
        if self._loaded_loras:
            await asyncio.to_thread(pipeline.unload_lora_weights)
            self._loaded_loras.clear()

        # Load new LoRAs
        if batch_lora_set:
            loaded = set()
            for filename in batch_lora_set:
                try:
                    await asyncio.to_thread(
                        pipeline.load_lora_weights,
                        settings.loras_dir,
                        weight_name=filename,
                        adapter_name=adapter_name_for[filename],
                    )
                    loaded.add(filename)
                except Exception as exc:
                    logger.error("Failed to load LoRA %s: %s", filename, exc)
                    # Skip this LoRA but continue with others
                    continue

            # Activate successfully loaded LoRAs with their weights
            adapter_names = [adapter_name_for[f] for f in loaded]
            weights = [batch_loras[f] for f in loaded]
            if adapter_names:
                await asyncio.to_thread(pipeline.enable_lora)
                await asyncio.to_thread(
                    pipeline.set_adapters, adapter_names, adapter_weights=weights
                )
            self._loaded_loras = loaded

    async def _apply_embedding(self, pipeline, embedding_path: str, name: str) -> None:
        """Load a .safetensors TI and inject its weights into the text encoder.

        Tokens are named ``name.0``, ``name.1``, … where *name* is the
        embedding file stem (e.g. ``my_dog``).
        """
        import torch

        try:
            from safetensors.torch import load_file
            embeds_dict = load_file(embedding_path)
            tensor = embeds_dict[list(embeds_dict.keys())[0]]  # [num_vectors, dim]

            tokenizer = pipeline.tokenizer
            text_encoder = pipeline.text_encoder
            num_vectors = tensor.shape[0]

            tokens = [f"{name}.{i}" for i in range(num_vectors)]

            # Only add tokens that aren't already present (idempotent for
            # repeated batches with the same embedding).
            existing = set(tokenizer.get_vocab().keys())
            new_tokens = [t for t in tokens if t not in existing]
            if new_tokens:
                tokenizer.add_tokens(new_tokens)
                text_encoder.resize_token_embeddings(len(tokenizer))

            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            with torch.no_grad():
                for i, tid in enumerate(token_ids):
                    text_encoder.get_input_embeddings().weight[tid] = tensor[i]

            logger.info("Applied TI embedding %r (%d vectors) from %s - mean weights %s", name, num_vectors, embedding_path, f'{tensor.mean(dim=1)}')
        except Exception as e:
            logger.error("Failed to apply TI embedding %s: %s", embedding_path, e)
            raise

    async def _resolve_prompt_embeddings(
        self, pipeline, prompts: list[str], neg_prompts: list[str]
    ) -> tuple[list[str], list[str]]:
        """Scan prompts for <name> tokens, load embeddings, rewrite prompts."""
        if self._embedding_service is None:
            return prompts, neg_prompts

        # 1. Collect all unique names from both prompt lists.
        all_text = " ".join(prompts + neg_prompts)
        names = set(EMBEDDING_RE.findall(all_text))
        if not names:
            return prompts, neg_prompts

        # 2. Load each embedding and patch the pipeline.
        loaded: dict[str, int] = {}  # name → num_vectors
        for name in names:
            entry = self._embedding_service.get(name)
            if entry is None:
                logger.warning("TI embedding %r not found in embeddings_dir", name)
                continue
            await self._apply_embedding(pipeline, entry.path, name)
            loaded[name] = entry.num_vectors

        if not loaded:
            return prompts, neg_prompts

        # 3. Rewrite prompts.
        return [expand_prompt(p, loaded) for p in prompts], [expand_prompt(p, loaded) for p in neg_prompts]

    async def _run_batch(self, batch: list[QueueItem]) -> bool:
        """Run a single batch: load model, generate, save, record.

        Returns True if the batch ended in an OOM (items have been requeued /
        failed by _handle_oom); False on success or non-OOM failure.
        """
        # Reset the cancel signal so this batch can run unless interrupted.
        self._cancel_event.clear()

        entry = self._registry.get(batch[0].params.model_key)
        if entry is None:
            error_msg = f"Model key {batch[0].params.model_key!r} not found in registry (have: {[e.key for e in self._registry.all()]})"
            for item in batch:
                await self._queue.mark_failed(item, error_msg)
                await self._broadcaster.broadcast_event(
                    "queue_item_failed",
                    {"item_id": item.item_id, "error": error_msg},
                )
            logger.error(error_msg)
            return False

        try:
            pipeline = await asyncio.to_thread(self._loader.load, entry, self.vae_slicing)
        except Exception as exc:
            for item in batch:
                await self._queue.mark_failed(item, str(exc))
                await self._broadcaster.broadcast_event(
                    "queue_item_failed",
                    {"item_id": item.item_id, "error": str(exc)},
                )
            logger.error("Pipeline load failed: %s", exc)
            return False

        self._loaded_model_key = entry.key

        # Swap scheduler based on sampler (not applicable to SANA — it uses its
        # own FlowMatchEulerDiscreteScheduler and we leave it untouched)
        is_sdxl: bool = entry.base == "sdxl"
        is_sana: bool = entry.base == "sana"
        preview_variant = "sdxl" if is_sdxl else ("sana" if is_sana else "sd1")

        #if not is_sana:
        # apply the scheduler, even on Sana
        await asyncio.to_thread(
            self._apply_scheduler,
            pipeline,
            batch[0].params.sampler,
            batch[0].params.prediction_type,
        )

        # Load and activate LoRAs for this batch
        await self._load_loras(pipeline, batch)

        # Scan prompts for <name> TI tokens, load & rewrite
        raw_prompts = [item.params.prompt for item in batch]
        raw_neg = [item.params.negative_prompt for item in batch]
        prompts, neg_prompts = await self._resolve_prompt_embeddings(
            pipeline, raw_prompts, raw_neg
        )
        for i, p in enumerate(prompts):
            if p != raw_prompts[i]:
                logging.info(f"prompt {i}: '{raw_prompts[i]}' -> '{p}'")
        for i, p in enumerate(neg_prompts):
            if p != raw_neg[i]:
                logging.info(f"prompt {i}: '{raw_neg[i]}' -> '{p}'")


        for item in batch:
            await self._queue.mark_processing(item)

        params = batch[0].params  # all share critical params
        seeds = [item.params.seed for item in batch]

        # Encode prompts.
        # When use_compel is ON (default), encode via Compel — this supports
        # weighted syntax and long prompts. CompelForSana handles SANA's Gemma
        # text encoder. When OFF, pass raw strings and let the pipeline encode
        # internally. Encoding runs in a thread because it calls the GPU-bound
        # text encoder.
        if entry.use_compel:
            if is_sana:
                prompt_kwargs: dict = await asyncio.to_thread(
                    build_sana_prompt_embeds, pipeline, prompts, neg_prompts
                )
            else:
                prompt_kwargs = await asyncio.to_thread(
                    build_prompt_embeds, pipeline, prompts, neg_prompts, is_sdxl=is_sdxl
                )
        else:
            prompt_kwargs = {
                "prompt": prompts if len(prompts) > 1 else prompts[0],
                "negative_prompt": neg_prompts if len(neg_prompts) > 1 else neg_prompts[0],
            }

        step_counter: dict[str, int] = {"n": 0}
        event_loop = asyncio.get_event_loop()

        def on_step(pipe, step: int, timestep, callback_kwargs: dict) -> dict:
            # Check for user-requested cancellation on every denoising step.
            if self._cancel_event.is_set():
                raise GenerationCancelled()

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
                                sample[i: i + 1], variant=preview_variant
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
                torch.Generator(device=settings.seed_device).manual_seed(s)
                for s in seeds
            ]
            if isinstance(pipeline.scheduler,
                          SDPipelineInferenceFlowMatchEulerDiscreteScheduler) and pipeline.scheduler.config.use_dynamic_shifting:
                image_pixel_count = params.width * params.height
                # for linear shift, we go from 1 to 3 over 1 megapixel
                assert pipeline.scheduler.config.time_shift_type == 'linear'
                shift = 1.0 + 0.5 * (image_pixel_count / 1024 ** 2)
                pipeline.scheduler.set_shift(shift)

            # Fast-path: if cancellation was requested before inference started,
            # skip straight to the GenerationCancelled handler.
            if self._cancel_event.is_set():
                raise GenerationCancelled()

            # Reset peak-memory counter immediately before the pipeline call so
            # that capture_peak() below reflects only this inference run.
            self._mem_advisor.reset_peak()

            # AMP autocast:
            # - SD1/SDXL: float16 autocast resolves mixed dtype mismatches.
            # - SANA: uses bfloat16 natively; float16 autocast would be wrong.
            # - CPU/MPS: no-op regardless.
            import contextlib
            if settings.device.startswith("cuda") and not is_sana:
                _device_type = "cuda"
                _autocast_ctx = torch.amp.autocast(device_type=_device_type, dtype=torch.float16)
            else:
                _autocast_ctx = contextlib.nullcontext()

            def _run_pipeline():
                with _autocast_ctx:
                    _step = 32 if is_sana else 8
                    w = max(256, round(params.width  / _step) * _step)
                    h = max(256, round(params.height / _step) * _step)
                    call_kwargs = dict(
                        **prompt_kwargs,
                        width=w,
                        height=h,
                        num_inference_steps=params.steps,
                        generator=generators,
                        callback_on_step_end=on_step,
                        callback_on_step_end_tensor_inputs=["latents"],
                    )
                    _img2img_cleanup = None
                    if params.init_image:
                        import PIL.Image as PILImage_
                        init_path = settings.images_dir / params.init_image
                        logger.warning("crop: init_path=%s exists=%s", init_path, init_path.exists())
                        init_pil = PILImage_.open(init_path).convert("RGB")
                        logger.warning("crop: init_pil loaded size=%s mode=%s", init_pil.size, init_pil.mode)
                        if params.crop_width > 0:
                            src_w, src_h = init_pil.size
                            out_w, out_h = params.width, params.height
                            logger.warning("crop: entering crop branch crop=(%s,%s,%s,%s) out=%sx%s src=%sx%s",
                                        params.crop_left, params.crop_top, params.crop_width, params.crop_height,
                                        out_w, out_h, src_w, src_h)
                            rough_w = min(params.crop_width * 1.2, 1.0)
                            rough_h = min(params.crop_height * 1.2, 1.0)
                            rough_l = max(params.crop_left - (rough_w - params.crop_width) / 2, 0.0)
                            rough_t = max(params.crop_top - (rough_h - params.crop_height) / 2, 0.0)
                            rough_l = min(rough_l, 1.0 - rough_w)
                            rough_t = min(rough_t, 1.0 - rough_h)
                            px_left = int(rough_l * src_w)
                            px_top = int(rough_t * src_h)
                            px_right = int((rough_l + rough_w) * src_w)
                            px_bottom = int((rough_t + rough_h) * src_h)
                            init_pil = init_pil.crop((px_left, px_top, px_right, px_bottom))
                            rough_out_w = int(out_w * 1.2)
                            rough_out_h = int(out_h * 1.2)
                            init_pil = init_pil.resize((rough_out_w, rough_out_h), PILImage.LANCZOS)
                            x0 = (rough_out_w - out_w) // 2
                            y0 = (rough_out_h - out_h) // 2
                            init_pil = init_pil.crop((x0, y0, x0 + out_w, y0 + out_h))
                            logger.warning("crop: after crop branch final size=%s", init_pil.size)
                        else:
                            init_pil = init_pil.resize((params.width, params.height), PILImage.LANCZOS)
                            logger.warning("crop: after direct resize final size=%s", init_pil.size)
                        if is_sana:
                            # Pre-compute truncated timesteps.
                            pipeline.scheduler.set_timesteps(params.steps, device=pipeline._execution_device)
                            init_t = min(int(params.steps * params.strength), params.steps)
                            t_start = max(params.steps - init_t, 0)
                            order = getattr(pipeline.scheduler, 'order', 1)
                            pipeline._img2img_timesteps = pipeline.scheduler.timesteps[t_start * order:]
                            pipeline._img2img_image = init_pil
                            logger.warning("crop: SANA img2img set _img2img_image=%s timesteps=%s",
                                        init_pil.size if init_pil else None,
                                        pipeline._img2img_timesteps.shape if hasattr(pipeline._img2img_timesteps, 'shape') else len(pipeline._img2img_timesteps))

                            # Patch scheduler.set_timesteps so that after
                            # retrieve_timesteps() generates the full schedule
                            # inside __call__, it immediately gets overwritten
                            # with our truncated schedule.
                            _orig_st = pipeline.scheduler.set_timesteps

                            def _patched_st(self, num_inference_steps=None, device=None, **kwargs):
                                _orig_st(num_inference_steps, device=device, **kwargs)
                                self.timesteps = pipeline._img2img_timesteps.to(
                                    device=device if device else self.timesteps.device)
                                self.sigmas = self.sigmas[-(len(self.timesteps) + 1):]  # keep sigmas aligned
                                pipeline._img2img_start_timestep = self.timesteps[:1]
                                self.set_begin_index(0)  # ← add this so scale_noise uses the right sigma

                            #def _patched_st(self, num_inference_steps=None, device=None, **kwargs):
                            #    _orig_st(num_inference_steps, device=device, **kwargs)
                            #    self.timesteps = pipeline._img2img_timesteps.to(device=device)
                            #    pipeline._img2img_start_timestep = self.timesteps[:1]
                            pipeline.scheduler.set_timesteps = types.MethodType(_patched_st, pipeline.scheduler)

                            def _img2img_cleanup():
                                pipeline._img2img_image = None
                                pipeline._img2img_start_timestep = None
                                if hasattr(pipeline, '_img2img_timesteps'):
                                    del pipeline._img2img_timesteps
                                pipeline.scheduler.set_timesteps = _orig_st
                        else:
                            call_kwargs['image'] = init_pil
                            call_kwargs['strength'] = params.strength
                    if type(pipeline) is SanaPAGPipeline:
                        call_kwargs['guidance_scale'] = 0
                        call_kwargs['pag_scale'] = params.cfg
                        call_kwargs['pag_adaptive_scale'] = params.cfg_rescale_multiplier
                    else:
                        call_kwargs['guidance_scale'] = params.cfg
                        if is_sana:
                            call_kwargs['use_resolution_binning'] = False
                            call_kwargs['complex_human_instruction'] = None
                        else:
                            # guidance_rescale is an SD1/SDXL concept;
                            # SanaPipeline doesn't accept this argument.
                            call_kwargs['guidance_rescale'] = params.cfg_rescale_multiplier
                    try:
                        #logger.warning(f"running pipeline with kwargs {call_kwargs}")
                        return pipeline(**call_kwargs)
                    finally:
                        if _img2img_cleanup is not None:
                            _img2img_cleanup()

            _t0 = time.perf_counter()
            result = await asyncio.to_thread(_run_pipeline)
            self._timing_tracker.record(time.perf_counter() - _t0, len(batch))

            # Capture peak memory and record the run in the advisor.
            peak_bytes = self._mem_advisor.capture_peak()
            pixel_area = params.width * params.height
            self._mem_advisor.record_success(pixel_area, len(batch), peak_bytes)

            for item, pil_image in zip(batch, result.images):
                image_name = await asyncio.to_thread(_save_image, pil_image)
                await self._queue.mark_completed(item, image_name)
                await self._broadcaster.broadcast_event(
                    "queue_item_completed",
                    {"item_id": item.item_id, "image_name": image_name},
                )
                await self._record_in_gallery(item, image_name, entry.name)

            return False  # success — no OOM

        except GenerationCancelled:
            logger.info("Generation cancelled by user — marking %d item(s) as cancelled.", len(batch))
            for item in batch:
                await self._queue.mark_cancelled(item)
                await self._broadcaster.broadcast_event(
                    "queue_item_cancelled",
                    {"item_id": item.item_id},
                )
            return False

        except Exception as exc:
            if _is_oom(exc):
                await self._handle_oom(batch, exc)
                return True
            else:
                logger.exception("Pipeline inference failed")
                for item in batch:
                    if item.status == ItemStatus.PROCESSING:
                        await self._queue.mark_failed(item, str(exc))
                        await self._broadcaster.broadcast_event(
                            "queue_item_failed",
                            {"item_id": item.item_id, "error": str(exc)},
                        )
                return False

    async def _handle_oom(self, batch: list[QueueItem], exc: Exception) -> None:
        """
        Respond to an out-of-memory error during inference.

        Strategy:
        - Free the VRAM cache immediately.
        - Notify the MemoryAdvisor so it regresses the probe batch size (halves it).
        - If the batch has more than one item, split it in half: immediately retry
          the first half and requeue the second half at the front of the queue.
        - If the batch is already a single item it cannot be split further;
          mark it as failed with a clear OOM message.
        """
        logger.exception("OOM during inference")  # log the actual traceback
        import torch
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if len(batch) <= 1:
            # Single item — cannot split further.
            self._mem_advisor.record_oom(1)
            logger.error("OOM on single-item batch — failing item %s.", batch[0].item_id)
            await self._queue.mark_failed(
                batch[0],
                "Out of memory — try reducing image size or number of steps.",
            )
            await self._broadcaster.broadcast_event(
                "queue_item_failed",
                {"item_id": batch[0].item_id,
                 "error": "Out of memory — try reducing image size or number of steps."},
            )
            return

        # Record OOM and regress the probe.
        self._mem_advisor.record_oom(len(batch))

        # Staggered fallback: an OOM at batch size >1 also force-enables VAE
        # slicing (batch-dimension decode split), so the halved retry below runs
        # with lower decode VRAM. Persisted so future loads keep it.
        self._force_enable_vae_slicing()

        # Split the batch in half.
        mid = max(1, len(batch) // 2)
        first_half = batch[:mid]
        second_half = batch[mid:]

        logger.warning(
            "OOM with batch size %d — halving: retrying %d now, requeueing %d.",
            len(batch), len(first_half), len(second_half),
        )

        # Requeue second half at front.
        await self._queue.requeue_items(second_half, prepend=True)

        # Immediately retry first half.
        _ = await self._run_batch(first_half)

    def _apply_scheduler(self, pipeline, sampler: str, prediction_type: str = 'default') -> None:
        """Swap the pipeline scheduler in-place based on the sampler name."""
        scheduler_class_name_and_kwargs = SAMPLER_MAP.get(sampler)
        if scheduler_class_name_and_kwargs is None:
            raise ValueError(f"Unknown sampler '{sampler}'")
        try:
            config = dict(pipeline.scheduler.config)
            scheduler_class_name, scheduler_kwargs = scheduler_class_name_and_kwargs
            if scheduler_class_name == "SDPipelineInferenceFlowMatchEulerDiscreteScheduler":
                scheduler_cls = SDPipelineInferenceFlowMatchEulerDiscreteScheduler
            else:
                scheduler_cls = getattr(diffusers, scheduler_class_name, None)
                if scheduler_cls is None:
                    raise ValueError(f"No scheduler in diffusers for class name '{scheduler_class_name}'")
            use_karras = sampler in KARRAS_SAMPLERS
            if use_karras:
                config["use_karras_sigmas"] = True
            # Apply prediction_type override when requested (skip for flow-match schedulers
            # which use their own step logic and don't accept this parameter).
            if (
                prediction_type
                and prediction_type != 'default'
                and scheduler_class_name != "SDPipelineInferenceFlowMatchEulerDiscreteScheduler"
            ):
                config["prediction_type"] = prediction_type
            config.update(**scheduler_kwargs)
            pipeline.scheduler = scheduler_cls.from_config(config, **scheduler_kwargs)
            print("scheduler prediction type:", getattr(pipeline.scheduler.config, "prediction_type", None))
        except Exception as e:
            logger.warning(f"Could not swap scheduler for sampler {sampler}: {repr(e)}")

    async def _record_in_gallery(
        self, item: QueueItem, image_name: str, model_name: str
    ) -> None:
        """Record a completed image in the gallery database and broadcast the event."""
        board_id = item.params.board_id or self._default_board_id
        if self._session_factory is None or board_id is None:
            return
        try:
            from backend.gallery.service import GalleryService
            async with self._session_factory() as session:
                service = GalleryService(session)
                record = await service.record_image(
                    item, image_name, board_id, model_name,
                    request_serial_number=item.item_id,
                )
                await session.commit()
                # Build event payload (same shape as _image_record_to_dict / _image_record_to_dict)
                created_at = record.created_at.strftime("%Y-%m-%d %H:%M:%S.%f") if record.created_at else None
                event_data = {
                    "image_name": image_name,
                    "board_id": board_id,
                    "prompt": item.params.prompt,
                    "negative_prompt": item.params.negative_prompt,
                    "seed": item.params.seed,
                    "width": item.params.width,
                    "height": item.params.height,
                    "steps": item.params.steps,
                    "cfg": item.params.cfg,
                    "cfg_rescale_multiplier": item.params.cfg_rescale_multiplier,
                    "sampler": item.params.sampler,
                    "prediction_type": item.params.prediction_type,
                    "model_key": item.params.model_key,
                    "model_name": model_name,
                    "lora_keys": item.params.lora_keys,
                    "lora_strengths": item.params.lora_strengths,
                    "starred": False,
                    "request_serial_number": record.request_serial_number,
                    "created_at": created_at,
                    "tags": [],
                }
            await self._broadcaster.broadcast_event("image_created", event_data)
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

