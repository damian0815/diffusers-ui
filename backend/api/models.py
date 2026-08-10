from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import asyncio

from backend.domain import ModelEntry
from backend.model_registry.registry import ModelNotFoundError, ModelMergeError
from backend.model_registry.loader import validate_merge_components
from backend.model_registry.scanner import scan_hf_cache, scan_local_dir

router = APIRouter(prefix="/api/v2/models", tags=["models"])


class InstallBody(BaseModel):
    name: str
    base: str
    source_type: str = "hf_repo_id"
    hf_api_key: str | None = None
    quantization: str | None = None


class UpdateBody(BaseModel):
    """Used by PATCH /{key} — all fields optional; only supplied ones are changed."""
    name: str | None = None
    quantization: str | None = None


class MergeComponentBody(BaseModel):
    key: str
    weight: float = 1.0


class MergeBody(BaseModel):
    name: str
    components: list[MergeComponentBody]


@router.get("/")
async def list_models(request: Request):
    """Return all registered models."""
    registry = request.app.state.registry
    models = registry.all()
    return {"models": [_entry_to_dict(m, registry) for m in models]}


@router.get("/get_by_attrs")
async def get_by_attrs(name: str, type: str, base: str, request: Request):
    """Find a registered model by attributes (InvokeAI-compatible)."""
    registry = request.app.state.registry
    for entry in registry.all():
        if entry.name == name and entry.base == base:
            return _entry_to_dict(entry, registry)
    raise HTTPException(status_code=404, detail=f"Model not found: name={name!r} base={base!r}")


@router.post("/merge")
async def create_merge(body: MergeBody, request: Request):
    """Create a merge model specification (no weights are produced).

    Validates component architectures at definition time. On mismatch returns
    400 with a structured detail so the UI can flag the offending model rows:
    {"detail": {"error": "...", "components": {key: reason}}}
    """
    registry = request.app.state.registry
    pairs = [(c.key, c.weight) for c in body.components]

    try:
        entries = registry.resolve_merge_components(pairs)
    except ModelMergeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        await asyncio.to_thread(validate_merge_components, entries)
    except ModelMergeError as exc:
        if exc.incompatible:
            raise HTTPException(
                status_code=400,
                detail={"error": str(exc), "components": exc.incompatible},
            )
        raise HTTPException(status_code=400, detail=str(exc))

    entry = registry.add_merge(body.name, pairs)
    return _entry_to_dict(entry, registry)


@router.post("/install")
async def install_model(source: str, body: InstallBody, request: Request):
    """Register a new model in the registry."""
    registry = request.app.state.registry
    entry = registry.register_new(
        name=body.name,
        base=body.base,
        source=source,
        source_type=body.source_type,
        hf_api_key=body.hf_api_key,
        quantization=body.quantization,
    )
    return _entry_to_dict(entry, registry)


@router.patch("/{key}")
async def update_model(key: str, body: UpdateBody, request: Request):
    """Update name and/or quantization for a registered model."""
    registry = request.app.state.registry
    try:
        entry = registry.get(key)
        if entry is None:
            raise ModelNotFoundError(f"No model with key {key!r}")
        if body.quantization is not None and entry.merge is not None:
            raise HTTPException(status_code=400, detail="Merge models do not support quantization.")
        if body.name is not None:
            entry = registry.rename(key, body.name)
        if body.quantization is not None or (
            # allow explicitly setting quantization to None via "" sentinel
            body.quantization == ""
        ):
            quant = body.quantization if body.quantization else None
            entry = registry.update_quantization(key, quant)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _entry_to_dict(entry, registry)


@router.delete("/{key}")
async def delete_model(key: str, request: Request):
    """Delete a model from the registry."""
    registry = request.app.state.registry
    try:
        registry.delete(key)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"deleted": key}


@router.get("/scan_hf_cache")
async def scan_hf_cache_endpoint(request: Request):
    """Return HF cache candidates. Does not modify the registry."""
    registry = request.app.state.registry
    candidates = scan_hf_cache()
    return {"candidates": [_entry_to_dict(c, registry) for c in candidates]}


@router.get("/scan_local_dir")
async def scan_local_dir_endpoint(path: str, request: Request):
    print("scanning", path)
    """Return local directory candidates. Does not modify the registry."""
    from pathlib import Path
    registry = request.app.state.registry
    candidates = scan_local_dir(Path(path))
    print("candidates", candidates)
    return {"candidates": [_entry_to_dict(c, registry) for c in candidates]}


@router.get("/loaded")
async def get_loaded_model(request: Request):
    """Return the key of the currently loaded model pipeline (null if none)."""
    loader = request.app.state.loader
    return {"loaded_key": loader.loaded_key}


@router.post("/unload_cache")
async def unload_model_cache(request: Request):
    """
    Unload the currently loaded model pipeline from memory/VRAM.

    Returns 409 if a generation is currently in progress (safe-guard —
    unloading while the pipeline is running would corrupt the inference).
    """
    worker = request.app.state.worker
    if not worker.is_idle:
        raise HTTPException(
            status_code=409,
            detail="Generation in progress — cannot unload model now. "
                   "Wait for the current job to finish.",
        )
    loader = request.app.state.loader
    if loader.loaded_key is None:
        return {"unloaded": False, "message": "No model was loaded."}
    await asyncio.to_thread(loader.unload)
    return {"unloaded": True}


def _entry_to_dict(entry: ModelEntry, registry) -> dict:
    files_missing = False
    if entry.merge is None and entry.source_type == "local_path":
        from pathlib import Path
        files_missing = not Path(entry.source).exists()
    return {
        "key": entry.key,
        "name": entry.name,
        "type": entry.base,   # legacy field kept for compat
        "base": entry.base,
        "source": entry.source,
        "source_type": entry.source_type,
        "kind": entry.kind,
        "has_api_key": entry.hf_api_key is not None,
        "quantization": entry.quantization,
        "files_missing": files_missing,
        "merge": (
            [
                {
                    "key": c.key,
                    "weight": c.weight,
                    "name": (registry.get(c.key).name if registry.get(c.key) else "(deleted)"),
                }
                for c in entry.merge
            ]
            if entry.merge
            else None
        ),
    }

