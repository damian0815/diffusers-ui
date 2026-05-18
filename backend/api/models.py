from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.domain import ModelEntry
from backend.model_registry.registry import ModelNotFoundError
from backend.model_registry.scanner import scan_hf_cache, scan_local_dir

router = APIRouter(prefix="/api/v2/models", tags=["models"])


class InstallBody(BaseModel):
    name: str
    base: str
    source_type: str = "hf_repo_id"
    hf_api_key: str | None = None


class RenameBody(BaseModel):
    name: str


@router.get("/")
async def list_models(request: Request):
    """Return all registered models."""
    registry = request.app.state.registry
    models = registry.all()
    return {"models": [_entry_to_dict(m) for m in models]}


@router.get("/get_by_attrs")
async def get_by_attrs(name: str, type: str, base: str, request: Request):
    """Find a registered model by attributes (InvokeAI-compatible)."""
    registry = request.app.state.registry
    for entry in registry.all():
        if entry.name == name and entry.base == base:
            return _entry_to_dict(entry)
    raise HTTPException(status_code=404, detail=f"Model not found: name={name!r} base={base!r}")


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
    )
    return _entry_to_dict(entry)


@router.patch("/{key}")
async def rename_model(key: str, body: RenameBody, request: Request):
    """Rename a registered model."""
    registry = request.app.state.registry
    try:
        entry = registry.rename(key, body.name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _entry_to_dict(entry)


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
    candidates = scan_hf_cache()
    return {"candidates": [_entry_to_dict(c) for c in candidates]}


@router.get("/scan_local_dir")
async def scan_local_dir_endpoint(path: str, request: Request):
    print("scanning", path)
    """Return local directory candidates. Does not modify the registry."""
    from pathlib import Path
    candidates = scan_local_dir(Path(path))
    print("candidates", candidates)
    return {"candidates": [_entry_to_dict(c) for c in candidates]}


def _entry_to_dict(entry: ModelEntry) -> dict:
    return {
        "key": entry.key,
        "name": entry.name,
        "base": entry.base,
        "source": entry.source,
        "source_type": entry.source_type,
        "has_api_key": entry.hf_api_key is not None,
    }

