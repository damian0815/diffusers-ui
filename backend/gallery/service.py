from __future__ import annotations

import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.domain import QueueItem
from backend.gallery.models import Gallery, ImageRecord, Tag


class GalleryNotFoundError(Exception):
    """Raised when a gallery (board) does not exist."""


class ImageNotFoundError(Exception):
    """Raised when an image record does not exist."""


class GalleryService:
    """
    All database operations. No SQL in route handlers — use this class.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    # ------------------------------------------------------------------
    # Galleries / Boards
    # ------------------------------------------------------------------

    async def list_galleries(self) -> list[Gallery]:
        result = await self._session.execute(select(Gallery).order_by(Gallery.created_at))
        return list(result.scalars().all())

    async def get_gallery(self, board_id: str) -> Gallery | None:
        result = await self._session.execute(
            select(Gallery).where(Gallery.board_id == board_id)
        )
        return result.scalar_one_or_none()

    async def create_gallery(self, name: str) -> Gallery:
        gallery = Gallery(board_id=str(uuid.uuid4()), name=name)
        self._session.add(gallery)
        await self._session.flush()
        await self._session.refresh(gallery)
        return gallery

    async def rename_gallery(self, board_id: str, new_name: str) -> Gallery:
        gallery = await self.get_gallery(board_id)
        if gallery is None:
            raise GalleryNotFoundError(f"Gallery {board_id!r} not found")
        gallery.name = new_name  # type: ignore[assignment]
        await self._session.flush()
        return gallery

    async def delete_gallery(
        self, board_id: str, include_images: bool = False
    ) -> None:
        gallery = await self.get_gallery(board_id)
        if gallery is None:
            raise GalleryNotFoundError(f"Gallery {board_id!r} not found")
        if include_images:
            result = await self._session.execute(
                select(ImageRecord).where(ImageRecord.gallery_id == gallery.id)
            )
            for img in result.scalars().all():
                await self._session.delete(img)
        await self._session.delete(gallery)
        await self._session.flush()

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    async def record_image(
        self,
        item: QueueItem,
        image_name: str,
        board_id: str,
        model_name: str,
        gen_seq: int | None = None,
    ) -> ImageRecord:
        """Persist a generated image record in the given gallery."""
        gallery = await self.get_gallery(board_id)
        p = item.params
        record = ImageRecord(
            image_name=image_name,
            gallery_id=gallery.id if gallery else None,
            prompt=p.prompt,
            negative_prompt=p.negative_prompt,
            seed=p.seed,
            width=p.width,
            height=p.height,
            steps=p.steps,
            cfg=p.cfg,
            cfg_rescale_multiplier=p.cfg_rescale_multiplier,
            sampler=p.sampler,
            model_key=p.model_key,
            model_name=model_name,
            gen_seq=gen_seq,
        )
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        return record

    async def get_image(self, image_name: str) -> ImageRecord | None:
        result = await self._session.execute(
            select(ImageRecord)
            .where(ImageRecord.image_name == image_name)
            .options(selectinload(ImageRecord.tags))
        )
        return result.scalar_one_or_none()

    async def count_images(
        self,
        board_id: str,
        search: str | None = None,
        starred_only: bool = False,
    ) -> int:
        """Return the total number of images matching the given filters."""
        gallery = await self.get_gallery(board_id)
        gallery_id = gallery.id if gallery else None

        query = select(func.count()).select_from(ImageRecord)
        if gallery_id is not None:
            query = query.where(ImageRecord.gallery_id == gallery_id)
        if search:
            pattern = f"%{search}%"
            query = query.where(
                or_(
                    ImageRecord.prompt.ilike(pattern),
                    ImageRecord.negative_prompt.ilike(pattern),
                )
            )
        if starred_only:
            query = query.where(ImageRecord.starred == True)  # noqa: E712

        result = await self._session.execute(query)
        return result.scalar_one()

    async def list_images(
        self,
        board_id: str,
        search: str | None = None,
        starred_only: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> list[ImageRecord]:
        """
        Paginated image list with optional full-text search across
        prompt and negative_prompt columns.
        """
        gallery = await self.get_gallery(board_id)
        gallery_id = gallery.id if gallery else None

        query = (
            select(ImageRecord)
            .options(selectinload(ImageRecord.tags))
            .order_by(ImageRecord.gen_seq.desc().nullslast(), ImageRecord.created_at.desc())
        )
        if gallery_id is not None:
            query = query.where(ImageRecord.gallery_id == gallery_id)
        if search:
            pattern = f"%{search}%"
            query = query.where(
                or_(
                    ImageRecord.prompt.ilike(pattern),
                    ImageRecord.negative_prompt.ilike(pattern),
                )
            )
        if starred_only:
            query = query.where(ImageRecord.starred == True)  # noqa: E712

        query = query.offset(offset).limit(limit)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def list_image_names(self, board_id: str) -> list[str]:
        gallery = await self.get_gallery(board_id)
        if gallery is None:
            return []
        result = await self._session.execute(
            select(ImageRecord.image_name)
            .where(ImageRecord.gallery_id == gallery.id)
            .order_by(ImageRecord.created_at.desc())
        )
        return list(result.scalars().all())

    async def move_images(self, image_names: list[str], target_board_id: str) -> int:
        """Move images to a different gallery. Returns count of moved images."""
        target = await self.get_gallery(target_board_id)
        if target is None:
            raise GalleryNotFoundError(f"Gallery {target_board_id!r} not found")
        count = 0
        for name in image_names:
            result = await self._session.execute(
                select(ImageRecord).where(ImageRecord.image_name == name)
            )
            record = result.scalar_one_or_none()
            if record is not None:
                record.gallery_id = target.id  # type: ignore[assignment]
                count += 1
        await self._session.flush()
        return count

    async def delete_image(self, image_name: str) -> str | None:
        """Delete an image record from the database. Returns the board_id it belonged to."""
        record = await self.get_image(image_name)
        if record is None:
            raise ImageNotFoundError(f"Image {image_name!r} not found")
        board_id: str | None = None
        if record.gallery_id is not None:
            from sqlalchemy import select as _select
            from backend.gallery.models import Gallery as _Gallery
            result = await self._session.execute(
                _select(_Gallery).where(_Gallery.id == record.gallery_id)
            )
            g = result.scalar_one_or_none()
            board_id = g.board_id if g else None
        await self._session.delete(record)
        await self._session.flush()
        return board_id

    # ------------------------------------------------------------------
    # Stars & Tags
    # ------------------------------------------------------------------

    async def set_starred(self, image_name: str, starred: bool) -> ImageRecord:
        record = await self.get_image(image_name)
        if record is None:
            raise ImageNotFoundError(f"Image {image_name!r} not found")
        record.starred = starred  # type: ignore[assignment]
        await self._session.flush()
        return record

    async def add_tag(self, image_name: str, tag_name: str) -> ImageRecord:
        record = await self.get_image(image_name)
        if record is None:
            raise ImageNotFoundError(f"Image {image_name!r} not found")
        # Get or create tag
        tag_result = await self._session.execute(
            select(Tag).where(Tag.name == tag_name)
        )
        tag = tag_result.scalar_one_or_none()
        if tag is None:
            tag = Tag(name=tag_name)
            self._session.add(tag)
            await self._session.flush()
        if tag not in record.tags:
            record.tags.append(tag)
        await self._session.flush()
        return record

    async def remove_tag(self, image_name: str, tag_name: str) -> ImageRecord:
        record = await self.get_image(image_name)
        if record is None:
            raise ImageNotFoundError(f"Image {image_name!r} not found")
        tag_result = await self._session.execute(
            select(Tag).where(Tag.name == tag_name)
        )
        tag = tag_result.scalar_one_or_none()
        if tag and tag in record.tags:
            record.tags.remove(tag)
        await self._session.flush()
        return record

