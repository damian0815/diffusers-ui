from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from backend.config import settings
from backend.gallery.db import get_session
from backend.gallery.service import GalleryService, GalleryNotFoundError, ImageNotFoundError

router = APIRouter(prefix="/api/v1/images", tags=["images"])


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
    session=Depends(get_session),
):
    """Return a paginated list of image records from a gallery."""
    service = GalleryService(session)
    records = await service.list_images(
        board_id=board_id,
        search=search,
        starred_only=starred_only,
        offset=offset,
        limit=limit,
    )
    return [_image_record_to_dict(r) for r in records]


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
        "cfg_rescale_multiplier": record.cfg_rescale_multiplier,
        "sampler": record.sampler,
        "model_key": record.model_key,
        "model_name": record.model_name,
        "starred": record.starred,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "tags": [t.name for t in record.tags],
    }

