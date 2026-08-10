"""LoRA discovery + install endpoints."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import settings
from backend.domain import LoRAEntry
from backend.model_registry.scanner import LORA_WEIGHTS_FILENAME, detect_lora_base

router = APIRouter(prefix="/api/v1/loras", tags=["loras"])

logger = logging.getLogger(__name__)

LORA_EXTENSIONS = {".safetensors", ".pt"}


class InstallLoraBody(BaseModel):
    source: str          # absolute path to a LoRA weight file or its folder
    name: str | None = None


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


@router.post("/install")
async def install_lora(body: InstallLoraBody) -> LoRAEntry:
    """
    Copy a LoRA weight file into the configured loras directory so the
    Generate panel can pick it up. Accepts either a weight file or a folder
    containing ``pytorch_lora_weights.safetensors``.
    """
    source_path = Path(body.source)
    if source_path.is_dir():
        source_path = source_path / LORA_WEIGHTS_FILENAME
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail=f"LoRA weights not found at {body.source}")

    is_marker_file = source_path.name == LORA_WEIGHTS_FILENAME
    default_name = source_path.parent.name if is_marker_file else source_path.stem
    name = (body.name or "").strip() or default_name

    loras_dir = settings.loras_dir
    loras_dir.mkdir(parents=True, exist_ok=True)
    target = (loras_dir / f"{name}{source_path.suffix.lower()}").resolve()
    source_resolved = source_path.resolve()

    if target != source_resolved and not target.exists():
        shutil.copy2(source_path, target)

    return LoRAEntry(
        name=target.stem,
        filename=target.name,
        path=str(target),
        base_models=[detect_lora_base(source_resolved)],
    )
