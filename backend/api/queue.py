from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.domain import ImageGenerationParams, QueueItem

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


class EnqueueRequest(BaseModel):
    prepend: bool = False
    batch: BatchParams


@router.post("/enqueue_batch")
async def enqueue_batch(queue_id: str, body: EnqueueRequest, request: Request):
    """
    Enqueue one generation job. Accepts the batch body format.
    Returns {"queue_id": str, "item_ids": [int]}.
    """
    queue = request.app.state.queue
    import random
    seed = body.batch.seed if body.batch.seed >= 0 else random.randint(0, 2**32 - 1)

    params = ImageGenerationParams(
        prompt=body.batch.prompt,
        negative_prompt=body.batch.negative_prompt,
        seed=seed,
        width=body.batch.width,
        height=body.batch.height,
        steps=body.batch.steps,
        cfg=body.batch.cfg,
        cfg_rescale_multiplier=body.batch.cfg_rescale_multiplier,
        sampler=body.batch.sampler,  # type: ignore[arg-type]
        model_key=body.batch.model_key,
    )
    item = QueueItem(params=params)
    await queue.enqueue(item, prepend=body.prepend)
    return {"queue_id": queue_id, "item_ids": [item.item_id]}


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

