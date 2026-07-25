import logging
from dataclasses import dataclass
from pathlib import Path

import safetensors.torch
import torch

from backend.config import settings

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingEntry:
    name: str          # filename stem, e.g. "my_dog"
    token: str         # placeholder token, e.g. "<sks>"
    path: str          # absolute path
    num_vectors: int   # how many vectors (1 = single-vector TI)
    embedding_dim: int # 768 for SD1, 1024 for SDXL

class EmbeddingService:
    """Manages learned embedding files in embeddings_dir."""

    def __init__(self):
        self._embeddings: dict[str, EmbeddingEntry] = {}

    def scan(self):
        """Scan embeddings_dir and populate the index."""
        settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings.clear()
        for path in settings.embeddings_dir.glob("*.safetensors"):
            try:
                meta = safetensors.torch.load_file(str(path))
                token_name = list(meta.keys())[0]
                tensor = meta[token_name]
                self._embeddings[path.stem] = EmbeddingEntry(
                    name=path.stem,
                    token=token_name,
                    path=str(path),
                    num_vectors=tensor.shape[0],
                    embedding_dim=tensor.shape[1],
                )
            except Exception as e:
                logger.warning(f"Failed to load embedding {path}: {e}")

    def all(self) -> list[EmbeddingEntry]:
        return list(self._embeddings.values())

    def get(self, name: str) -> EmbeddingEntry | None:
        return self._embeddings.get(name)

    def rescan(self):
        """Re-read embeddings_dir (e.g. after a training run saves a new file)."""
        self.scan()

    def delete(self, name: str) -> bool:
        entry = self._embeddings.pop(name, None)
        if entry:
            Path(entry.path).unlink(missing_ok=True)
            return True
        return False
