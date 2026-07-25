"""LoRA discovery endpoint."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter

from backend.config import settings
from backend.domain import LoRAEntry

router = APIRouter(prefix="/api/v1/loras", tags=["loras"])

logger = logging.getLogger(__name__)

LORA_EXTENSIONS = {".safetensors", ".pt"}


@router.get("/")
async def list_loras() -> list[LoRAEntry]:
    """Scan the configured loras directory for LoRA weight files."""
    loras_dir = settings.loras_dir
    entries: list[LoRAEntry] = []

    if not loras_dir.exists():
        return entries

    for path in sorted(loras_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in LORA_EXTENSIONS:
            name = path.stem  # filename without extension
            entries.append(LoRAEntry(
                name=name,
                filename=path.name,
                path=str(path.resolve()),
                base_models=[],  # TODO: detect from tensor key prefixes
            ))

    return entries
