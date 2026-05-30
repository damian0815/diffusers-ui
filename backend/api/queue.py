from __future__ import annotations

import logging
import random
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, ValidationError

from backend.domain import ImageGenerationParams, QueueItem
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/queue/{queue_id}", tags=["queue"])


class BatchParams(BaseModel):
    """Parameters for a single generation request."""
    prompt: str
    negative_prompt: str = ""
    seed: int = -1
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg: float = 7.5
    cfg_rescale_multiplier: float = 0.0
    sampler: str = "euler_a"
    model_key: str
    board_id: str  # target gallery board; None = worker default


class EnqueueRequest(BaseModel):
    prepend: bool = False
    batch: BatchParams


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_request_format(body: dict) -> str:
    """Return ``'clean'`` for the native flat format, ``'annoying'`` for the
    legacy InvokeAI graph format.

    The distinguishing feature is whether ``body["batch"]`` contains a
    ``"graph"`` key — the InvokeAI wire format always does.
    """
    batch = body.get("batch", {})
    if isinstance(batch, dict) and "graph" in batch:
        return "annoying"
    return "clean"


# ---------------------------------------------------------------------------
# Parser helpers (shared by _parse_enqueue_request_annoying)
# ---------------------------------------------------------------------------

def _find_node(nodes: dict, *names: str) -> dict | None:
    """Return the first node whose dict-key exactly matches one of *names*,
    or whose key starts with ``'<name>:'`` (SDXL colon-suffixed style).
    Tries names in order; returns ``None`` if nothing matches.
    """
    for name in names:
        if name in nodes:
            return nodes[name]
        for k, v in nodes.items():
            if k.startswith(name + ":"):
                return v
    return None


def _from_data_field(data: list, field_name: str) -> Any:
    """Scan the InvokeAI ``batch.data`` array for the first entry whose
    ``field_name`` matches and return its first item value, or ``None``.
    """
    for group in data:
        for entry in group:
            if entry.get("field_name") == field_name:
                items = entry.get("items", [])
                if items:
                    return items[0]
    return None


def _extract_model_key(model_data: Any) -> str:
    """Extract a plain model-key string from whatever the client put in the
    model-loader node's ``model`` field.

    Handles:
    * a plain string (already the key)
    * a dict — tries ``key``, ``model_key``, ``name`` in that order
    """
    if isinstance(model_data, str):
        return model_data
    if isinstance(model_data, dict):
        for field in ("key", "model_key", "name"):
            if model_data.get(field):
                return str(model_data[field])
    return str(model_data or "")


# ---------------------------------------------------------------------------
# Per-format parsers
# Precondition:  body is a raw dict parsed from the JSON request body.
# Postcondition: returns (BatchParams, prepend: bool); raises HTTPException
#                on unrecoverable parse errors.
# ---------------------------------------------------------------------------

def _parse_enqueue_request_clean(body: dict) -> tuple[BatchParams, bool]:
    """Parse the native flat ``{prepend, batch: {…}}`` format via Pydantic."""
    try:
        req = EnqueueRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())
    return req.batch, req.prepend


def _parse_enqueue_request_annoying(body: dict) -> tuple[BatchParams, bool]:
    """Parse the legacy InvokeAI graph-format request.

    Supports both SD-1/SD-2 (exact node names such as ``sd1_model_loader``,
    ``noise``, ``pos_cond``) and SDXL (colon-suffixed names such as
    ``noise:0``, ``denoise_latents:1``).

    Extraction strategy:
    * Seed and positive prompt are read from ``batch.data`` first because that
      array is always populated and its structure is consistent across both
      model types.
    * All other fields (negative prompt, dimensions, sampler settings,
      model key, board id) are read directly from graph nodes because they
      are never batched and are always present in the node graph.
    """
    prepend = bool(body.get("prepend", False))
    batch   = body.get("batch", {})
    nodes: dict = batch.get("graph", {}).get("nodes", {})
    data:  list = batch.get("data", [])

    # Locate nodes — works for both exact and colon-suffixed names
    model_node   = _find_node(nodes, "sdxl_model_loader", "sd1_model_loader")
    noise_node   = _find_node(nodes, "noise")
    denoise_node = _find_node(nodes, "denoise_latents")
    neg_node     = _find_node(nodes, "neg_cond")
    pos_node     = _find_node(nodes, "pos_cond", "positive_prompt")
    canvas_node  = _find_node(nodes, "canvas_output")

    # Seed — data array is authoritative; fall back to the noise node
    raw_seed = _from_data_field(data, "seed")
    seed = (
        int(raw_seed)
        if raw_seed is not None
        else int((noise_node or {}).get("seed", -1))
    )

    # Positive prompt — prefer data (works for both SD1 and SDXL); fall
    # back to the pos_cond / positive_prompt node
    raw_prompt = (
        _from_data_field(data, "positive_prompt")
        or _from_data_field(data, "prompt")
    )
    if raw_prompt is None and pos_node:
        raw_prompt = (
            pos_node.get("prompt")
            or pos_node.get("text")
            or pos_node.get("value")
        )
    prompt = str(raw_prompt or "")

    # Fields only in nodes (never batched)
    negative_prompt       = str((neg_node     or {}).get("prompt", ""))
    width                 = int((noise_node   or {}).get("width",  512))
    height                = int((noise_node   or {}).get("height", 512))
    steps                 = int((denoise_node or {}).get("steps",  20))
    cfg                   = float((denoise_node or {}).get("cfg_scale",             7.5))
    cfg_rescale_multiplier = float((denoise_node or {}).get("cfg_rescale_multiplier", 0.0))
    sampler               = str((denoise_node or {}).get("scheduler", "euler_a"))
    board_id              = str(((canvas_node or {}).get("board") or {}).get("board_id"))
    model_key             = _extract_model_key((model_node or {}).get("model"))

    try:
        batch_params = BatchParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            cfg_rescale_multiplier=cfg_rescale_multiplier,
            sampler=sampler,
            model_key=model_key,
            board_id=board_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse graph-format request: {exc}",
        )

    return batch_params, prepend


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.post("/enqueue_batch")
async def enqueue_batch(queue_id: str, request: Request, body: dict = Body(...)):
    """
    Enqueue one generation job.

    Accepts both the native flat format and the legacy InvokeAI graph format
    (auto-detected via the presence of ``batch.graph``).
    Returns {"queue_id": str, "items": [<image-record-dict>]}.
    """
    fmt = _detect_request_format(body)
    if fmt == "annoying":
        batch, prepend = _parse_enqueue_request_annoying(body)
    else:
        batch, prepend = _parse_enqueue_request_clean(body)

    #logger.info(f"enqueue_batch ({prepend=}) {batch}")

    queue    = request.app.state.queue
    registry = request.app.state.registry
    seed     = batch.seed if batch.seed >= 0 else random.randint(0, 2**32 - 1)

    if batch.board_id is None:
        raise ValueError("Missing board id")

    params = ImageGenerationParams(
        prompt=batch.prompt,
        negative_prompt=batch.negative_prompt,
        seed=seed,
        width=batch.width,
        height=batch.height,
        steps=batch.steps,
        cfg=batch.cfg,
        cfg_rescale_multiplier=batch.cfg_rescale_multiplier,
        sampler=batch.sampler,  # type: ignore[arg-type]
        model_key=batch.model_key,
        board_id=batch.board_id,
    )
    item = QueueItem(params=params)
    await queue.enqueue(item, prepend=prepend)

    model_entry = registry.get(params.model_key)
    model_name  = model_entry.name if model_entry else None
    item_dict   = _queue_item_to_image_dict(item, model_name)

    broadcaster = request.app.state.broadcaster
    await broadcaster.broadcast_event("queue_item_enqueued", item_dict)

    return {"queue_id": queue_id, "items": [item_dict]}


@router.get("/i/{item_id}")
async def get_queue_item(queue_id: str, item_id: int, request: Request):
    """
    Returns item status and output image name if completed.
    """
    queue = request.app.state.queue
    item = queue.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Queue item {item_id} not found")
    return {
        "queue_id": queue_id,
        "item_id": item.item_id,
        "status": item.status.value,
        "output_image_name": item.output_image_name,
        "error": item.error,
    }


@router.delete("/i/{item_id}")
async def cancel_item(queue_id: str, item_id: int, request: Request):
    """Cancel a pending queue item."""
    queue = request.app.state.queue
    item = queue.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Queue item {item_id} not found")
    cancelled = await queue.cancel_item(item_id)
    return {"queue_id": queue_id, "item_id": item_id, "cancelled": cancelled}


@router.put("/clear")
async def clear_queue(queue_id: str, request: Request):
    """Cancel all pending items in the queue."""
    queue = request.app.state.queue
    count = await queue.cancel_all()
    return {"queue_id": queue_id, "cancelled_count": count}


@router.put("/prune")
async def prune_queue(queue_id: str, request: Request):
    """Remove completed/failed/cancelled items from tracking."""
    queue = request.app.state.queue
    count = await queue.prune()
    return {"queue_id": queue_id, "pruned_count": count}


@router.get("/")
async def list_queue(queue_id: str, request: Request):
    """List all tracked queue items."""
    queue = request.app.state.queue
    items = queue.all_items()
    return {
        "queue_id": queue_id,
        "items": [
            {
                "item_id": item.item_id,
                "status": item.status.value,
                "output_image_name": item.output_image_name,
                "error": item.error,
                "prompt": item.params.prompt,
                "model_key": item.params.model_key,
            }
            for item in items
        ],
    }


@router.post("/i/{item_id}/resubmit")
async def resubmit_item(queue_id: str, item_id: int, request: Request, prepend: bool = True):
    """
    Create a fresh clone of an existing queue item and enqueue it.
    Works on any item, regardless of status (useful for failed/cancelled/completed jobs).
    Returns the new item_id.
    """
    queue = request.app.state.queue
    item = queue.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Queue item {item_id} not found")
    new_item = QueueItem(params=item.params)
    await queue.enqueue(new_item, prepend=prepend)

    registry    = request.app.state.registry
    model_entry = registry.get(new_item.params.model_key)
    model_name  = model_entry.name if model_entry else None
    item_dict   = _queue_item_to_image_dict(new_item, model_name)

    broadcaster = request.app.state.broadcaster
    await broadcaster.broadcast_event("queue_item_enqueued", item_dict)

    return {"queue_id": queue_id, "item_id": new_item.item_id, "resubmitted_from": item_id}


@router.put("/i/{item_id}/move_front")
async def move_item_to_front(queue_id: str, item_id: int, request: Request):
    """Move a pending queue item to the front of the queue."""
    queue = request.app.state.queue
    moved = await queue.move_to_front(item_id)
    if not moved:
        raise HTTPException(
            status_code=400,
            detail=f"Item {item_id} is not pending or not found in queue"
        )
    return {"queue_id": queue_id, "item_id": item_id, "moved": True}


def _queue_item_to_image_dict(item: QueueItem, model_name: str | None) -> dict:
    """
    Return an image-record-shaped dict for a QueueItem.
    Fields that are only known after generation (image_name, gen_seq,
    created_at, starred, tags) are returned as None / empty defaults.
    Extra queue fields (item_id, status) are included for caller convenience.
    """
    p = item.params
    return {
        "item_id": item.item_id,
        "status": item.status.value,
        "image_name": item.output_image_name,
        "prompt": p.prompt,
        "negative_prompt": p.negative_prompt,
        "seed": p.seed,
        "width": p.width,
        "height": p.height,
        "steps": p.steps,
        "cfg": p.cfg,
        "cfg_rescale_multiplier": p.cfg_rescale_multiplier,
        "sampler": p.sampler,
        "model_key": p.model_key,
        "model_name": model_name,
        "board_id": p.board_id,
        "starred": False,
        "gen_seq": None,
        "created_at": None,
        "tags": [],
    }
