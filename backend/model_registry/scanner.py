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
                        base=_guess_base(repo.repo_id),
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
                    base=_guess_base(child.name),
                    source=str(child.resolve()),
                    source_type="local_path",
                )
            )
    return candidates


def _looks_like_pipeline(path: Path) -> bool:
    """Return True if path contains model_index.json."""
    return (path / "model_index.json").exists()


def _guess_base(name: str) -> str:
    """Heuristically guess the model base from the name/repo-id."""
    name_lower = name.lower()
    if "xl" in name_lower or "sdxl" in name_lower:
        return "sdxl"
    if "2.0" in name_lower or "2-" in name_lower or "sd2" in name_lower:
        return "sd-2"
    return "sd-1"

