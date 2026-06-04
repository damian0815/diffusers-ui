from __future__ import annotations

import gc
import logging

from backend.config import settings
from backend.domain import ModelEntry

# Sub-module names that hold the bulk of VRAM across all pipeline types.
_HEAVY_SUBMODULES = (
    "transformer", "unet",
    "text_encoder", "text_encoder_2", "text_encoder_3",
    "vae", "image_encoder",
)


class PipelineLoadError(Exception):
    """Raised when a pipeline fails to load."""


class PipelineLoader:
    """
    Loads and unloads a single DiffusionPipeline.
    Only one model is loaded at a time (memory-efficient for local use).
    """

    def __init__(self):
        self._pipeline = None
        self._loaded_key: str | None = None

    def load(self, entry: ModelEntry):
        """Load the pipeline for the given model entry. No-op if already loaded."""
        if self._loaded_key == entry.key:
            return self._pipeline
        self._unload()
        try:
            if entry.base == "sana":
                self._pipeline = _load_sana_pipeline(entry)
            else:
                from diffusers import DiffusionPipeline
                kwargs: dict = {"torch_dtype": _dtype_for_device(settings.device)}
                if entry.hf_api_key:
                    kwargs["token"] = entry.hf_api_key
                logging.info(f"Loading model {entry.name} from {entry.source} onto {settings.device} @{settings.dtype}")
                self._pipeline = DiffusionPipeline.from_pretrained(
                    entry.source, **kwargs
                ).to(device=settings.device)
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
                for name in _HEAVY_SUBMODULES:
                    mod = getattr(pipeline, name, None)
                    if mod is not None:
                        remove_hook_from_module(mod, recurse=True)
            except Exception:
                pass  # accelerate not installed, or no hooks — fine either way

            # ── Step 2: move sub-modules to CPU ─────────────────────────
            # Calling .to("cpu") replaces each CUDA tensor with a CPU copy
            # and releases the CUDA allocation *immediately*, before GC runs.
            # This is the most reliable way to free VRAM on model switch.
            if is_cuda:
                for name in _HEAVY_SUBMODULES:
                    mod = getattr(pipeline, name, None)
                    if mod is None:
                        continue
                    try:
                        mod.to("cpu")
                    except Exception:
                        pass  # quantised modules may not support .to()

            # ── Step 3: release Python object and run GC ─────────────────
            del pipeline
            gc.collect()

            # ── Step 4: return freed blocks to the CUDA allocator ────────
            if is_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception:
            logging.exception("Error during pipeline unload — VRAM may not be fully freed")

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

    return pipeline


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


