from __future__ import annotations

import json
from pathlib import Path

from backend.domain import ModelEntry


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
    Walk `directory` looking for subdirectories that contain a
    `model_index.json` (diffusers pipeline marker).
    Returns ModelEntry candidates. Does NOT write to the registry.
    """
    if not directory.is_dir():
        return []

    candidates: list[ModelEntry] = []
    for child in directory.iterdir():
        if child.is_dir() and _looks_like_pipeline(child):
            candidates.append(
                ModelEntry(
                    key=str(child.resolve()),
                    name=child.name,
                    base=_detect_base(child),
                    source=str(child.resolve()),
                    source_type="local_path",
                )
            )
    return candidates


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

