from __future__ import annotations

import logging

import torch

from backend.config import settings
from backend.domain import ModelEntry


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
            from diffusers import DiffusionPipeline
            kwargs: dict = {"torch_dtype": _dtype_for_device(settings.device)}
            if entry.hf_api_key:
                kwargs["token"] = entry.hf_api_key
            logging.info(f"Loading model {entry.name} from {entry.source} onto {settings.device} @{settings.dtype}")
            self._pipeline = DiffusionPipeline.from_pretrained(
                entry.source, **kwargs
            ).to(device=settings.device, dtype=settings.dtype)
            _enable_fast_attention(self._pipeline)
            self._loaded_key = entry.key
        except Exception as exc:
            raise PipelineLoadError(
                f"Failed to load model {entry.name!r} from {entry.source!r}: {exc}"
            ) from exc
        return self._pipeline

    def _unload(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded_key = None
            # Free GPU memory if applicable
            if settings.device == "cuda":
                import torch
                torch.cuda.empty_cache()

    @property
    def loaded_key(self) -> str | None:
        return self._loaded_key


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


