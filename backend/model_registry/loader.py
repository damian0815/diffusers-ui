from __future__ import annotations

import gc
import logging
import types
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization",
)

import torch
from diffusers import SanaPAGPipeline
from diffusers.utils.torch_utils import randn_tensor

from backend.config import settings
from backend.domain import ModelEntry

class PipelineLoadError(Exception):
    """Raised when a pipeline fails to load."""


class PipelineLoader:
    """
    Loads and unloads a single DiffusionPipeline.
    Only one model is loaded at a time (memory-efficient for local use).
    """

    def __init__(self, registry=None):
        self._pipeline = None
        self._loaded_key: str | None = None
        self._registry = registry   # ModelRegistry, used to resolve merge components

    def load(self, entry: ModelEntry):
        """Load the pipeline for the given model entry. No-op if already loaded."""
        if self._loaded_key == entry.key:
            return self._pipeline
        self._unload()
        try:
            if entry.merge:
                self._pipeline = _load_merge_pipeline(entry, self._registry)
            elif entry.base == "sana":
                self._pipeline = _load_sana_pipeline(entry)
            else:
                from diffusers import DiffusionPipeline
                kwargs: dict = {"torch_dtype": _dtype_for_device(settings.device)}
                if entry.hf_api_key:
                    kwargs["token"] = entry.hf_api_key
                logging.info(f"Loading model {entry.name} from {entry.source} onto {settings.device} @{settings.dtype}")
                self._pipeline = DiffusionPipeline.from_pretrained(
                    entry.source, **kwargs
                ).to(device=settings.device, dtype=settings.dtype)
                self._pipeline.vae = self._pipeline.vae.to(dtype=settings.vae_dtype)
                _enable_fast_attention(self._pipeline)
            self._loaded_key = entry.key
        except Exception as exc:
            raise PipelineLoadError(
                f"Failed to load model {entry.name!r} from {entry.source!r}: {exc}"
            ) from exc
        return self._pipeline

    def _unload(self) -> None:
        if self._pipeline is None:
            return
        logging.info("Unloading pipeline (key=%s)", self._loaded_key)
        pipeline = self._pipeline
        self._pipeline = None
        self._loaded_key = None

        try:
            import torch
            is_cuda = settings.device.startswith("cuda")

            # ── Step 1: strip accelerate device-map hooks ────────────────
            # device_map pipelines (e.g. quantised SANA) register forward
            # hooks on every sub-module that hold strong references.  Removing
            # them lets the GC collect the modules once we del the pipeline.
            try:
                from accelerate.hooks import remove_hook_from_module
                for _name, mod in pipeline.components.items():
                    remove_hook_from_module(mod, recurse=True)
            except Exception:
                pass  # accelerate not installed, or no hooks — fine either way

            # ── Step 2: move every sub-module to CPU ────────────────────
            # Calling .to("cpu") replaces each CUDA tensor with a CPU copy
            # and releases the CUDA allocation *immediately*, before GC runs.
            # We iterate over ALL named children (not just a known subset)
            # so that components like the scheduler, safety_checker,
            # controlnet, etc. are covered.  Each child is wrapped in
            # try/except so quantised modules (e.g. bitsandbytes 4-bit)
            # that cannot be moved to CPU fail gracefully without aborting.
            if is_cuda:
                for _name, mod in pipeline.components.items():
                    try:
                        mod.to("cpu")
                    except Exception:
                        pass

            # ── Step 3: release Python object and run GC ─────────────────
            del pipeline
            gc.collect()

            # ── Step 4: synchronise then return freed blocks ────────────
            # synchronize() must come BEFORE empty_cache() so any pending
            # CUDA operations complete first, otherwise the allocator may
            # keep blocks alive that reference in-flight work.
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        except Exception:
            logging.exception("Error during pipeline unload — VRAM may not be fully freed")

    def unload(self) -> None:
        """Public API: unload the current pipeline and free VRAM."""
        self._unload()

    @property
    def loaded_key(self) -> str | None:
        return self._loaded_key


# ---------------------------------------------------------------------------
# SANA-specific loader
# ---------------------------------------------------------------------------

def _load_sana_pipeline(entry: ModelEntry):
    """
    Load a SanaPipeline with optional bitsandbytes quantization.

    SANA dtype requirements:
      - The full pipeline is loaded in float32 so the VAE stays stable.
      - text_encoder and transformer are then cast to bfloat16.
    With quantization (int8 / int4):
      - Both text_encoder and transformer are loaded with BitsAndBytesConfig.
      - device_map="balanced" handles placement automatically (no manual .to()).
    """
    import torch
    from diffusers import SanaPipeline

    quant = entry.quantization  # None | 'int8' | 'int4'
    logging.info(
        "Loading SANA model %r from %s (quantization=%s)",
        entry.name, entry.source, quant or "none",
    )

    if quant in ("int8", "int4"):
        try:
            from transformers import BitsAndBytesConfig as TransformersBnBConfig, AutoModel
            from diffusers import BitsAndBytesConfig as DiffusersBnBConfig, SanaTransformer2DModel
        except ImportError as exc:
            raise PipelineLoadError(
                "bitsandbytes is required for quantized SANA loading. "
                "Install it with: pip install bitsandbytes"
            ) from exc

        load_in_8bit = quant == "int8"
        load_in_4bit = quant == "int4"

        bnb_kwargs_te = dict(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        bnb_kwargs_tr = dict(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)

        te_quant_cfg = TransformersBnBConfig(**bnb_kwargs_te)
        text_encoder = AutoModel.from_pretrained(
            entry.source,
            subfolder="text_encoder",
            quantization_config=te_quant_cfg,
            torch_dtype=torch.bfloat16,
            token=entry.hf_api_key,
        )

        tr_quant_cfg = DiffusersBnBConfig(**bnb_kwargs_tr)
        transformer = SanaTransformer2DModel.from_pretrained(
            entry.source,
            subfolder="transformer",
            quantization_config=tr_quant_cfg,
            torch_dtype=torch.bfloat16,
            token=entry.hf_api_key,
        )

        pipeline = SanaPipeline.from_pretrained(
            entry.source,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            token=entry.hf_api_key,
        )
    else:
        # Standard (non-quantized) SANA loading
        load_kwargs: dict = {"torch_dtype": torch.float32}
        if entry.hf_api_key:
            load_kwargs["token"] = entry.hf_api_key
        pipeline = SanaPipeline.from_pretrained(entry.source, **load_kwargs)
        pipeline.to(device=settings.device)
        # Cast the heavy compute modules to bfloat16 to halve VRAM usage
        pipeline.text_encoder.to(torch.bfloat16)
        pipeline.transformer = pipeline.transformer.to(torch.bfloat16)

    # ── img2img monkey patches ──────────────────────────────────────
    _apply_sana_img2img_patches(pipeline)

    return pipeline


# ---------------------------------------------------------------------------
# img2img monkey-patches for SanaPipeline
# ---------------------------------------------------------------------------

def _apply_sana_img2img_patches(pipe):
    """Monkey-patch a SanaPipeline instance to support img2img.

    Only ``prepare_latents`` is patched (a regular method — instance
    resolution works).  The worker is responsible for setting
    ``_img2img_image`` on the pipe and patching ``scheduler.set_timesteps``
    to truncate the denoising schedule.
    """
    _orig_prepare_latents = pipe.prepare_latents

    def patched_prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        image = getattr(self, "_img2img_image", None)
        if image is None or latents is not None:
            return _orig_prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents)

        image_t = self.image_processor.preprocess(image, height=height, width=width).to(device=device, dtype=self.vae.dtype)
        with torch.no_grad():
            image_latents = self.vae.encode(image_t).latent
        image_latents = image_latents * self.vae.config.scaling_factor
        if image_latents.shape[0] != batch_size:
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=dtype)

        # 1. Use scale_noise, not add_noise
        # 2. After _patched_st fires, self.scheduler.timesteps[0] IS the start timestep
        self.scheduler.set_begin_index(0)
        return self.scheduler.scale_noise(
            sample=image_latents.to(dtype),
            timestep=self.scheduler.timesteps[:1].expand(batch_size),
            noise=noise,
        )
        #start_t = self._img2img_start_timestep.to(device).repeat(batch_size)
        #return self.scheduler.add_noise(image_latents.to(dtype), noise, start_t)

    pipe.prepare_latents = types.MethodType(patched_prepare_latents, pipe)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dtype_for_device(device: str):
    """Return the appropriate torch dtype for the given device."""
    import torch
    if device == "cpu":
        return torch.float32
    return torch.float16


def _enable_fast_attention(pipeline) -> None:
    """
    Attempt to enable the fastest available attention implementation.

    Priority order:
    1. xformers memory-efficient attention (explicit install, fastest on Ampere)
    2. PyTorch 2.0+ native SDPA / Flash Attention (automatic when using torch >= 2.0)
    3. Fall back to standard attention silently.
    """
    # xformers — fastest for float16 on Ampere (RTX 3090 etc.)
    try:
        import xformers  # noqa: F401
        pipeline.enable_xformers_memory_efficient_attention()
        logging.info("xformers memory-efficient attention enabled.")
        return
    except Exception as exc:
        logging.info("xformers not available (%s); trying PyTorch SDPA.", exc)

    # PyTorch 2.x+ SDPA — automatically uses Flash Attention 2 when available.
    # Just enabling the flag here; diffusers uses it via the attention_processor.
    try:
        import torch
        if hasattr(torch.backends, "cuda") and int(torch.__version__.split(".")[0]) >= 2:
            # attempt_enable_mem_efficient_sdp / enable_flash_sdp are best-effort
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logging.info("PyTorch SDPA (Flash Attention) enabled.")
    except Exception as exc:
        logging.info("Could not configure PyTorch SDPA: %s", exc)


# ---------------------------------------------------------------------------
# Model merge helpers
# ---------------------------------------------------------------------------

def _resolve_transformer_sd_paths(entry: ModelEntry) -> list:
    """Return the path(s) of a component's transformer weights.

    - local_path: read straight from the model dir.
    - hf_repo_id: snapshot_download restricted to the transformer subfolder
      (downloads only if not cached; token supports private repos).
    Handles both a single safetensors file and the sharded
    ``diffusion_pytorch_model-00001-of-00002.safetensors`` layout.
    """
    from pathlib import Path
    if entry.source_type == "local_path":
        root = Path(entry.source)
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(
            repo_id=entry.source,
            allow_patterns=["transformer/*"],
            token=entry.hf_api_key,
        ))
    transformer_dir = root / "transformer"
    single = transformer_dir / "diffusion_pytorch_model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(transformer_dir.glob("diffusion_pytorch_model-*.safetensors"))
    if shards:
        return shards
    raise PipelineLoadError(
        f"No transformer weights found under {transformer_dir!s} "
        f"(expected diffusion_pytorch_model.safetensors or shards)."
    )


def _close_safetensors_handles(handles) -> None:
    """Close safe_open handles if supported; newer safetensors versions
    free the underlying mmap when the handle is garbage collected."""
    for h in handles:
        close = getattr(h, "close", None)
        if close is not None:
            close()


def validate_merge_components(components: list[ModelEntry]) -> None:
    """Architecture pre-check: all components must expose the SAME key set.

    Compares every component's transformer key set against the first (base)
    component. Raises ModelMergeError with `incompatible` = {component_key: reason}
    for each mismatching component. Must run in a thread (may download HF
    weights on first use).
    """
    from backend.model_registry.registry import ModelMergeError
    from safetensors import safe_open

    def _key_set(entry: ModelEntry) -> set:
        handles = [
            safe_open(str(p), framework="pt", device="cpu")
            for p in _resolve_transformer_sd_paths(entry)
        ]
        try:
            return set().union(*(set(h.keys()) for h in handles))
        finally:
            _close_safetensors_handles(handles)

    base = components[0]
    base_keys = _key_set(base)
    incompatible: dict[str, str] = {}
    for comp in components[1:]:
        keys = _key_set(comp)
        if keys != base_keys:
            incompatible[comp.key] = (
                f"'{comp.name}' has a different architecture to the base model "
                f"'{base.name}' ({len(keys)} transformer weight keys vs "
                f"{len(base_keys)}). Mixing architectures (e.g. 600M + 1600M) "
                f"is not supported."
            )
    if incompatible:
        raise ModelMergeError("Merge architecture mismatch", incompatible=incompatible)


def _merge_transformer_state_dicts(
    components: list[ModelEntry], weights: list[float]
) -> dict:
    """Blend the transformer state dicts of ``components`` in CPU RAM.

    For each key: weighted sum over all components, accumulating in the dtype
    of the first tensor encountered (native dtype). Alphas are normalized over
    the sum of all weights. Closes all safetensors handles (mmaps) on exit.
    """
    from safetensors import safe_open

    def _open_component(entry: ModelEntry):
        handles = [
            safe_open(str(p), framework="pt", device="cpu")
            for p in _resolve_transformer_sd_paths(entry)
        ]
        key_sets = [set(h.keys()) for h in handles]     # cache per-handle
        return handles, set().union(*key_sets), key_sets

    component_data = [_open_component(e) for e in components]
    try:
        total = sum(weights)
        alphas = [w / total for w in weights]
        ref_dtype = None
        merged: dict[str, torch.Tensor] = {}

        for key in component_data[0][1]:
            acc = None
            for (handles, _, key_sets), alpha in zip(component_data, alphas):
                idx = next(i for i, ks in enumerate(key_sets) if key in ks)
                t = handles[idx].get_tensor(key)
                if ref_dtype is None:
                    ref_dtype = t.dtype
                t = t.to(ref_dtype)
                acc = t * alpha if acc is None else acc + t * alpha
            merged[key] = acc

        return merged
    finally:
        for handles, _, _ in component_data:
            _close_safetensors_handles(handles)


def _load_merge_pipeline(entry: ModelEntry, registry) -> object:
    """Load a merge spec: blend transformer SDs, load base pipeline, swap weights."""
    if registry is None:
        raise PipelineLoadError("PipelineLoader was created without a registry; cannot resolve merge.")
    components = []
    for spec in entry.merge:
        comp = registry.get(spec.key)
        if comp is None:
            raise PipelineLoadError(
                f"Merge {entry.name!r} references model {spec.key!r} which is no longer in the registry."
            )
        components.append(comp)

    weights = [spec.weight for spec in entry.merge]
    merged_sd = _merge_transformer_state_dicts(components, weights)

    pipeline = _load_sana_pipeline(entry)          # base pipeline == first component
    try:
        pipeline.transformer.load_state_dict(merged_sd)   # strict=True (default)
    finally:
        del merged_sd                                 # free the CPU RAM merge buffer

    return pipeline


