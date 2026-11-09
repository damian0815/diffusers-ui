from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.gallery.db import get_session
from backend.gallery.service import GalleryService, GalleryNotFoundError, IMG2IMG_BOARD_NAME

router = APIRouter(prefix="/api/v1/boards", tags=["boards"])

logger = logging.getLogger(__name__)

@router.get("/")
async def list_boards(session=Depends(get_session)):
    """Return all galleries (boards)."""
    service = GalleryService(session)
    galleries = await service.list_galleries()
    dicts = []
    for g in galleries:
        count = await service.count_images_in_gallery(g.board_id)
        latest = await service.get_latest_image_in_gallery(g.board_id)
        dicts.append(_gallery_to_dict(g, count=count, latest_image=latest))
    #logger.warning(f"list_boards() returning dicts: {dicts}")
    return dicts


@router.post("/")
async def create_board(board_name: str, request: Request, session=Depends(get_session)):
    """Create a new gallery."""
    service = GalleryService(session)
    gallery = await service.create_gallery(board_name)
    d = _gallery_to_dict(gallery)
    await request.app.state.broadcaster.broadcast_event("board_created", d)
    return d


@router.patch("/{board_id}")
async def rename_board(
    board_id: str, board_name: str, request: Request, session=Depends(get_session)
):
    """Rename an existing gallery."""
    service = GalleryService(session)
    try:
        gallery = await service.rename_gallery(board_id, board_name)
    except GalleryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    d = _gallery_to_dict(gallery)
    await request.app.state.broadcaster.broadcast_event("board_renamed", d)
    return d


@router.delete("/{board_id}")
async def delete_board(
    board_id: str,
    request: Request,
    include_images: bool = False,
    session=Depends(get_session),
):
    """Delete a gallery, optionally including all its images."""
    service = GalleryService(session)

    # Protect the img2img source-image board — it stores files needed for
    # restoring old img2img generations via adoptSettings('all').
    gallery = await service.get_gallery(board_id)
    if gallery and gallery.name == IMG2IMG_BOARD_NAME:
        raise HTTPException(
            status_code=403,
            detail=f"Board {IMG2IMG_BOARD_NAME!r} cannot be deleted because it stores source images "
                   f"needed for restoring old img2img generations.  Delete the images within it first.",
        )

    deleted_images = []
    try:
        deleted_images = await service.delete_gallery(board_id, include_images=include_images)
    except GalleryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    await request.app.state.broadcaster.broadcast_event(
        "board_deleted", {"board_id": board_id}
    )
    return {"deleted": board_id, "deleted_images": deleted_images}


def _gallery_to_dict(gallery, count: int=None, latest_image: str=None) -> dict:
    d = {
        "board_id": gallery.board_id,
        "board_name": gallery.name,
        "created_at": gallery.created_at.strftime("%Y-%m-%d %H:%M:%S.%f") if gallery.created_at else None,
    }
    if count is not None:
        d["image_count"] = count
    if latest_image is not None:
        d["latest_image"] = latest_image
        d["latest_image_name"] = latest_image
    return d
