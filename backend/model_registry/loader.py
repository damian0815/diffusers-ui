from __future__ import annotations

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
            self._pipeline = DiffusionPipeline.from_pretrained(
                entry.source, **kwargs
            ).to(settings.device)
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

