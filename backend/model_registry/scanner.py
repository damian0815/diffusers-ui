from __future__ import annotations

import json
import os
from pathlib import Path

from backend.domain import ModelEntry

LORA_WEIGHTS_FILENAME = "pytorch_lora_weights.safetensors"


def scan_hf_cache() -> list[ModelEntry]:
    """
    Scan ~/.cache/huggingface/hub and return a ModelEntry candidate
    for each repo that looks like a diffusers pipeline.
    Does NOT write to the registry — just returns candidates.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return []

    candidates: list[ModelEntry] = []
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return []

    for repo in cache_info.repos:
        if repo.repo_type != "model":
            continue
        # Check if any snapshot contains a model_index.json (diffusers pipeline marker)
        for revision in repo.revisions:
            snapshot_path = revision.snapshot_path
            if snapshot_path and _looks_like_pipeline(Path(snapshot_path)):
                candidates.append(
                    ModelEntry(
                        key=repo.repo_id.replace("/", "--"),
                        name=repo.repo_id.split("/")[-1],
                        base=_detect_base(Path(snapshot_path)),
                        source=repo.repo_id,
                        source_type="hf_repo_id",
                    )
                )
                break  # one candidate per repo is enough

    return candidates


def scan_local_dir(directory: Path) -> list[ModelEntry]:
    """
    Recursively walk ``directory`` for two kinds of candidates:

    - diffusers pipelines, marked by a ``model_index.json`` file
    - LoRA folders, marked by a ``pytorch_lora_weights.safetensors`` file

    Returns ModelEntry candidates. Does NOT write to the registry.
    """
    if not directory.is_dir():
        return []

    candidates: list[ModelEntry] = []
    for root, dirs, _files in os.walk(directory):
        root_path = Path(root)
        if _looks_like_pipeline(root_path):
            candidates.append(_pipeline_entry(root_path))
            # Don't descend into the pipeline's submodules (unet/, vae/, …)
            dirs[:] = []
            continue
        lora_file = root_path / LORA_WEIGHTS_FILENAME
        if lora_file.is_file():
            candidates.append(_lora_entry(root_path, lora_file))
            # Don't descend into the LoRA folder's internals
            dirs[:] = []
            continue
    return candidates


def _pipeline_entry(pipeline_dir: Path) -> ModelEntry:
    """Build a ModelEntry candidate for a diffusers pipeline directory."""
    return ModelEntry(
        key=str(pipeline_dir.resolve()),
        name=pipeline_dir.name,
        base=_detect_base(pipeline_dir),
        source=str(pipeline_dir.resolve()),
        source_type="local_path",
    )


def _lora_entry(lora_dir: Path, weights_file: Path) -> ModelEntry:
    """Build a ModelEntry candidate for a LoRA folder."""
    return ModelEntry(
        key=str(weights_file.resolve()),
        name=lora_dir.name,
        base=detect_lora_base(weights_file),
        source=str(lora_dir.resolve()),
        source_type="local_path",
        kind="lora",
    )


def detect_lora_base(weights_file: Path) -> str:
    """
    Best-effort LoRA base detection from the safetensors key prefixes.
    Falls back to a name-based guess when the header can't be read.
    """
    try:
        from safetensors import safe_open
        with safe_open(str(weights_file), framework="pt") as f:
            keys = list(f.keys())
    except Exception:
        return _guess_base_from_name(weights_file.parent.name)

    if any("lora_transformer" in k for k in keys):
        return "sana"
    if any("lora_te1" in k or "lora_te2" in k for k in keys):
        return "sdxl"
    if any("lora_te_text_model" in k for k in keys):
        return "sd-1"
    return _guess_base_from_name(weights_file.parent.name)


def _looks_like_pipeline(path: Path) -> bool:
    """Return True if path contains model_index.json."""
    return (path / "model_index.json").exists()


def _detect_base(pipeline_dir: Path) -> str:
    """
    Determine the model base by inspecting the pipeline's ``model_index.json``.
    Falls back to heuristic name-based guessing if the file can't be read.
    """
    index_path = pipeline_dir / "model_index.json"
    try:
        with open(index_path) as f:
            index = json.load(f)
    except Exception:
        return _guess_base_from_name(pipeline_dir.name)

    # SANA pipelines use a transformer instead of a UNet
    if "transformer" in index:
        return "sana"

    # SDXL has two text encoders; SD1/2 have only one
    if "text_encoder_2" in index:
        return "sdxl"

    # SD1 vs SD2: check the unet config's cross_attention_dim
    try:
        unet_cfg_path = pipeline_dir / "unet" / "config.json"
        with open(unet_cfg_path) as f:
            unet_cfg = json.load(f)
        # SD2 uses 1024-dim cross attention; SD1 uses 768
        if unet_cfg.get("cross_attention_dim") == 1024:
            return "sd-2"
    except Exception:
        pass

    return "sd-1"


def _guess_base_from_name(name: str) -> str:
    """Fallback heuristic: guess the model base from the name/repo-id."""
    name_lower = name.lower()
    if "sana" in name_lower:
        return "sana"
    if "xl" in name_lower or "sdxl" in name_lower:
        return "sdxl"
    if "2.0" in name_lower or "2-" in name_lower or "sd2" in name_lower:
        return "sd-2"
    return "sd-1"

