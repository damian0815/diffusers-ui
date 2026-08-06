"""Unit tests for backend/gallery/service.py"""
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from backend.gallery.db import Base
import backend.gallery.models  # noqa: F401
from backend.gallery.service import GalleryService, GalleryNotFoundError, ImageNotFoundError
from backend.domain import QueueItem, make_params


@pytest.fixture
async def service():
    """In-memory SQLite service instance — isolated per test."""
    from sqlalchemy.pool import StaticPool
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield GalleryService(session)
        await session.rollback()
    await engine.dispose()


def make_item(**overrides):
    return QueueItem(params=make_params(**overrides))


# ------------------------------------------------------------------
# Gallery CRUD
# ------------------------------------------------------------------

async def test_create_and_list_gallery(service):
    await service.create_gallery("Test Board")
    galleries = await service.list_galleries()
    assert any(g.name == "Test Board" for g in galleries)


async def test_get_gallery_by_board_id(service):
    g = await service.create_gallery("My Board")
    fetched = await service.get_gallery(g.board_id)
    assert fetched is not None
    assert fetched.name == "My Board"


async def test_get_nonexistent_gallery_returns_none(service):
    result = await service.get_gallery("does-not-exist")
    assert result is None


async def test_rename_gallery(service):
    g = await service.create_gallery("Old Name")
    await service.rename_gallery(g.board_id, "New Name")
    fetched = await service.get_gallery(g.board_id)
    assert fetched.name == "New Name"


async def test_rename_nonexistent_gallery_raises(service):
    with pytest.raises(GalleryNotFoundError):
        await service.rename_gallery("bad-id", "X")


async def test_delete_gallery(service):
    g = await service.create_gallery("To Delete")
    await service.delete_gallery(g.board_id)
    assert await service.get_gallery(g.board_id) is None


async def test_delete_nonexistent_gallery_raises(service):
    with pytest.raises(GalleryNotFoundError):
        await service.delete_gallery("bad-id")


# ------------------------------------------------------------------
# Image CRUD
# ------------------------------------------------------------------

async def test_record_image(service):
    g = await service.create_gallery("Board")
    item = make_item()
    record = await service.record_image(item, "img1.png", g.board_id, "my-model",
                                        request_serial_number=item.item_id)
    assert record.image_name == "img1.png"
    assert record.model_name == "my-model"
    assert record.prompt == item.params.prompt
    assert record.request_serial_number == item.item_id


async def test_get_image(service):
    g = await service.create_gallery("Board")
    item = make_item()
    await service.record_image(item, "img2.png", g.board_id, "model")
    record = await service.get_image("img2.png")
    assert record is not None
    assert record.image_name == "img2.png"


async def test_get_nonexistent_image_returns_none(service):
    result = await service.get_image("nope.png")
    assert result is None


async def test_list_images_basic(service):
    g = await service.create_gallery("Board")
    for i in range(3):
        await service.record_image(make_item(), f"img{i}.png", g.board_id, "model")
    images = await service.list_images(g.board_id)
    assert len(images) == 3


async def test_list_images_ordered_by_request_serial_number(service):
    """
    Images recorded with explicit request_serial_number values must be
    returned in request_serial_number descending order regardless of the order
    they were recorded in.  This mirrors the real scenario where batch
    processing reorders items: items enqueued as A(1), B(2), C(3), D(4), E(5)
    may be recorded in batch order [A, C, D] then [B, E], but the gallery
    query must sort by request_serial_number DESC so that the full set reads
    as E, D, C, B, A (i.e. reverse-enqueue, which the frontend flips back to
    enqueue order).
    """
    g = await service.create_gallery("Board")

    # Simulate batch 1: process A(1), C(3), D(4) first
    await service.record_image(make_item(seed=1), "A.png", g.board_id, "model", request_serial_number=1)
    await service.record_image(make_item(seed=3), "C.png", g.board_id, "model", request_serial_number=3)
    await service.record_image(make_item(seed=4), "D.png", g.board_id, "model", request_serial_number=4)

    # Simulate batch 2: process B(2), E(5) later
    await service.record_image(make_item(seed=2), "B.png", g.board_id, "model", request_serial_number=2)
    await service.record_image(make_item(seed=5), "E.png", g.board_id, "model", request_serial_number=5)

    images = await service.list_images(g.board_id)
    names = [img.image_name for img in images]
    # Expected: request_serial_number DESC → 5(E), 4(D), 3(C), 2(B), 1(A)
    assert names == ["E.png", "D.png", "C.png", "B.png", "A.png"], f"got {names}"


async def test_list_images_none_request_serial_number_last(service):
    """
    Images whose request_serial_number is NULL must appear after any image
    with a non-NULL value, sorted by created_at DESC among themselves.
    """
    g = await service.create_gallery("Board")

    # Record two images with explicit serials
    await service.record_image(make_item(seed=1), "A.png", g.board_id, "model", request_serial_number=10)
    await service.record_image(make_item(seed=2), "B.png", g.board_id, "model", request_serial_number=20)

    # Record one image WITHOUT a serial (NULL)
    await service.record_image(make_item(seed=3), "C.png", g.board_id, "model")

    images = await service.list_images(g.board_id)
    names = [img.image_name for img in images]
    # request_serial_number DESC: B(20), A(10), then C(NULL) last
    assert names == ["B.png", "A.png", "C.png"], f"got {names}"


async def test_search_by_prompt(service):
    g = await service.create_gallery("Board")
    await service.record_image(make_item(prompt="ocean sunset"), "img1.png", g.board_id, "model")
    await service.record_image(make_item(prompt="mountain fog"), "img2.png", g.board_id, "model")
    results = await service.list_images(g.board_id, search="ocean")
    assert len(results) == 1
    assert results[0].image_name == "img1.png"


async def test_list_image_names(service):
    g = await service.create_gallery("Board")
    await service.record_image(make_item(), "img_a.png", g.board_id, "model")
    await service.record_image(make_item(), "img_b.png", g.board_id, "model")
    names = await service.list_image_names(g.board_id)
    assert "img_a.png" in names
    assert "img_b.png" in names


async def test_pagination(service):
    g = await service.create_gallery("Board")
    for i in range(10):
        await service.record_image(make_item(), f"img{i:02d}.png", g.board_id, "model")
    page1 = await service.list_images(g.board_id, offset=0, limit=5)
    page2 = await service.list_images(g.board_id, offset=5, limit=5)
    assert len(page1) == 5
    assert len(page2) == 5
    names_p1 = {i.image_name for i in page1}
    names_p2 = {i.image_name for i in page2}
    assert names_p1.isdisjoint(names_p2)


# ------------------------------------------------------------------
# Stars & Tags
# ------------------------------------------------------------------

async def test_star_image(service):
    g = await service.create_gallery("Board")
    await service.record_image(make_item(), "img.png", g.board_id, "model")
    record = await service.set_starred("img.png", True)
    assert record.starred is True
    record = await service.set_starred("img.png", False)
    assert record.starred is False


async def test_star_nonexistent_raises(service):
    with pytest.raises(ImageNotFoundError):
        await service.set_starred("nope.png", True)


async def test_starred_only_filter(service):
    g = await service.create_gallery("Board")
    await service.record_image(make_item(), "starred.png", g.board_id, "model")
    await service.record_image(make_item(), "plain.png", g.board_id, "model")
    await service.set_starred("starred.png", True)
    results = await service.list_images(g.board_id, starred_only=True)
    assert len(results) == 1
    assert results[0].image_name == "starred.png"


async def test_add_and_remove_tag(service):
    g = await service.create_gallery("Board")
    await service.record_image(make_item(), "img.png", g.board_id, "model")
    record = await service.add_tag("img.png", "nature")
    assert any(t.name == "nature" for t in record.tags)
    record = await service.remove_tag("img.png", "nature")
    assert not any(t.name == "nature" for t in record.tags)


async def test_add_tag_to_nonexistent_raises(service):
    with pytest.raises(ImageNotFoundError):
        await service.add_tag("nope.png", "tag")


async def test_tag_shared_across_images(service):
    g = await service.create_gallery("Board")
    await service.record_image(make_item(), "img1.png", g.board_id, "model")
    await service.record_image(make_item(), "img2.png", g.board_id, "model")
    await service.add_tag("img1.png", "shared")
    await service.add_tag("img2.png", "shared")
    r1 = await service.get_image("img1.png")
    r2 = await service.get_image("img2.png")
    assert any(t.name == "shared" for t in r1.tags)
    assert any(t.name == "shared" for t in r2.tags)


