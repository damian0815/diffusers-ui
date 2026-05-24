from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.gallery.db import get_session
from backend.gallery.service import GalleryService, GalleryNotFoundError

router = APIRouter(prefix="/api/v1/boards", tags=["boards"])


@router.get("/")
async def list_boards(session=Depends(get_session)):
    """Return all galleries (boards)."""
    service = GalleryService(session)
    galleries = await service.list_galleries()
    return [_gallery_to_dict(g) for g in galleries]


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
    try:
        await service.delete_gallery(board_id, include_images=include_images)
    except GalleryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    await request.app.state.broadcaster.broadcast_event(
        "board_deleted", {"board_id": board_id}
    )
    return {"deleted": board_id}


def _gallery_to_dict(gallery) -> dict:
    return {
        "board_id": gallery.board_id,
        "board_name": gallery.name,
        "created_at": gallery.created_at.isoformat() if gallery.created_at else None,
    }
