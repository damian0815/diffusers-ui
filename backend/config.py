from dataclasses import dataclass, field
from pathlib import Path

import torch


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    images_dir: Path = field(default_factory=lambda: Path("images"))
    thumbnails_dir: Path = field(default_factory=lambda: Path("images/thumbs"))
    models_json: Path = field(default_factory=lambda: Path("models.json"))
    db_url: str = "sqlite+aiosqlite:///gallery.db"
    thumbnail_size: tuple[int, int] = (256, 256)
    # Output format for generated images: "jpg" (much smaller on disk, lossy)
    # or "png" (lossless). Existing .png records keep working either way.
    image_format: str = "jpg"
    jpeg_quality: int = 95
    batch_window_s: float = 0.3   # how long to wait collecting items before batching
    loras_dir: Path = field(default_factory=lambda: Path("loras"))
    embeddings_dir: Path = field(default_factory=lambda: Path("embeddings"))
    preview_every_n_steps: int = 1
    # Preview method: "taesd" (higher quality, uses tiny neural decoder) or
    # "approx" (linear RGB projection — fast, no extra weights required).
    preview_method: str = "taesd"
    device: str = field(default_factory=_detect_device)
    seed_device: str = 'cpu'

    @property
    def dtype(self) -> torch.dtype:
        if self.device == "cuda":
            # float16 is required for Flash Attention 2 on Ampere/Ada GPUs (e.g. RTX 3090)
            return torch.bfloat16
        elif self.device == "mps":
            return torch.float32
        elif self.device == "cpu":
            return torch.float32
        else:
            raise NotImplementedError(f"Unhandled device {self.device}")

    @property
    def vae_dtype(self) -> torch.dtype:
        return torch.float32


# Singleton — import this everywhere
settings = Settings()

