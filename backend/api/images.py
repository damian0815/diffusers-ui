from __future__ import annotations

import asyncio
import json
from io import BytesIO
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image as PILImage
from pydantic import BaseModel

from backend.config import settings
from backend.gallery.db import get_session
from backend.gallery.service import GalleryService, GalleryNotFoundError, ImageNotFoundError

router = APIRouter(prefix="/api/v1/images", tags=["images"])


class MoveImagesRequest(BaseModel):
    image_names: list[str]
    source_board_id: str
    target_board_id: str


@router.post("/move")
async def move_images(body: MoveImagesRequest, request: Request, session=Depends(get_session)):
    """Move a set of images to a different gallery."""
    service = GalleryService(session)
    try:
        count = await service.move_images(body.image_names, body.target_board_id)
    except GalleryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    await request.app.state.broadcaster.broadcast_event("images_moved", {
        "image_names": body.image_names,
        "source_board_id": body.source_board_id,
        "target_board_id": body.target_board_id,
    })
    return {"moved": count, "target_board_id": body.target_board_id}


@router.get("/names")
async def list_image_names(board_id: str, session=Depends(get_session)):
    """Return all image names in a gallery."""
    service = GalleryService(session)
    names = await service.list_image_names(board_id)
    return {"image_names": names}


@router.get("/i/{image_name}/full")
async def get_image_full(image_name: str):
    """Serve the raw image file."""
    path = settings.images_dir / image_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image {image_name!r} not found")
    return FileResponse(str(path), media_type="image/png")


@router.get("/i/{image_name}/thumbnail")
async def get_thumbnail(image_name: str):
    """Serve the thumbnail image."""
    path = settings.thumbnails_dir / image_name
    if not path.exists():
        # Fall back to the full image if thumbnail is missing
        path = settings.images_dir / image_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Thumbnail for {image_name!r} not found")
    return FileResponse(str(path), media_type="image/png")


@router.get("/i/{image_name}/metadata")
async def get_image_metadata(image_name: str, session=Depends(get_session)):
    """Return the stored metadata for an image."""
    service = GalleryService(session)
    record = await service.get_image(image_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Image {image_name!r} not found")
    return _image_record_to_dict(record)


@router.get("/i/{image_name}/gallery")
async def get_image_gallery(image_name: str, session=Depends(get_session)):
    """Return the gallery (board) info for an image."""
    service = GalleryService(session)
    record = await service.get_image_with_gallery(image_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Image {image_name!r} not found")
    gallery = record.gallery
    return {"board_id": gallery.board_id if gallery else None, "board_name": gallery.name if gallery else None}


@router.delete("/i/{image_name}")
async def delete_image(image_name: str, request: Request, session=Depends(get_session)):
    """Delete an image record and its files from disk."""
    service = GalleryService(session)
    try:
        board_id = await service.delete_image(image_name)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    await request.app.state.broadcaster.broadcast_event(
        "image_deleted", {"image_name": image_name, "board_id": board_id}
    )
    return {"deleted": image_name}


@router.patch("/i/{image_name}/star")
async def set_star(image_name: str, starred: bool, session=Depends(get_session)):
    """Toggle the starred flag on an image."""
    service = GalleryService(session)
    try:
        record = await service.set_starred(image_name, starred)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _image_record_to_dict(record)


@router.post("/i/{image_name}/tags/{tag_name}")
async def add_tag(image_name: str, tag_name: str, session=Depends(get_session)):
    """Add a tag to an image."""
    service = GalleryService(session)
    try:
        record = await service.add_tag(image_name, tag_name)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _image_record_to_dict(record)


@router.delete("/i/{image_name}/tags/{tag_name}")
async def remove_tag(image_name: str, tag_name: str, session=Depends(get_session)):
    """Remove a tag from an image."""
    service = GalleryService(session)
    try:
        record = await service.remove_tag(image_name, tag_name)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _image_record_to_dict(record)

@router.get("/")
async def list_images(
    board_id: str,
    search: str | None = None,
    starred_only: bool = False,
    offset: int = 0,
    limit: int = 50,
    sort_by: str = "request_serial_number",
    sort_dir: str = "desc",
    starred_first: bool = False,
    session=Depends(get_session),
):
    """Return a paginated list of image records plus the total matching count."""
    service = GalleryService(session)
    records, total = await asyncio.gather(
        service.list_images(
            board_id=board_id,
            search=search,
            starred_only=starred_only,
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            sort_dir=sort_dir,
            starred_first=starred_first,
        ),
        service.count_images(
            board_id=board_id,
            search=search,
            starred_only=starred_only,
        ),
    )
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [_image_record_to_dict(r) for r in records],
    }

@router.post("/upload-source")
async def upload_source_image(file: UploadFile = File(...)):
    """Accept an uploaded image for img2img; save and return its metadata."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    name = f"{uuid.uuid4().hex}.png"
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.images_dir / name
    pil_image = PILImage.open(BytesIO(image_bytes))
    pil_image.save(out_path)
    width, height = pil_image.size
    return {"image_name": name, "width": width, "height": height}


def _image_record_to_dict(record) -> dict:
    return {
        "image_name": record.image_name,
        "prompt": record.prompt,
        "negative_prompt": record.negative_prompt,
        "seed": record.seed,
        "width": record.width,
        "height": record.height,
        "steps": record.steps,
        "cfg": record.cfg,
        "cfg_scale": record.cfg, # duplicate, for annoying client
        "cfg_rescale_multiplier": record.cfg_rescale_multiplier,
        "sampler": record.sampler,
        "scheduler": record.sampler, # duplicate, for annoying client
        "prediction_type": getattr(record, 'prediction_type', 'default') or 'default',
        "model_key": record.model_key,
        "model_name": record.model_name,
        "starred": record.starred,
        "request_serial_number": record.request_serial_number,
        "gen_seq": record.request_serial_number,  # legacy alias
        "created_at": record.created_at.strftime("%Y-%m-%d %H:%M:%S.%f") if record.created_at else None,
        "tags": [t.name for t in record.tags],
        "source_image": record.source_image,
        "source_width": record.source_width,
        "source_height": record.source_height,
        "strength": record.strength,
        "crop_left": record.crop_left,
        "crop_top": record.crop_top,
        "crop_width": record.crop_width,
        "crop_height": record.crop_height,
        "lora_keys": json.loads(record.lora_keys) if record.lora_keys else [],
        "lora_strengths": json.loads(record.lora_strengths) if record.lora_strengths else [],
    }

