"""Integration tests for /api/v1/images/* endpoints."""
import pytest
from pathlib import Path
from PIL import Image as PILImage


async def create_board(client, name="Test Board") -> str:
    resp = await client.post(f"/api/v1/boards/?board_name={name}")
    assert resp.status_code == 200
    return resp.json()["board_id"]


@pytest.mark.asyncio
async def test_list_images_empty_board(client):
    board_id = await create_board(client)
    resp = await client.get(f"/api/v1/images/?board_id={board_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["items"], list)
    assert len(data["items"]) == 0


@pytest.mark.asyncio
async def test_list_image_names_empty(client):
    board_id = await create_board(client)
    resp = await client.get(f"/api/v1/images/names?board_id={board_id}")
    assert resp.status_code == 200
    assert "image_names" in resp.json()
    assert resp.json()["image_names"] == []


@pytest.mark.asyncio
async def test_get_nonexistent_image_full_returns_404(client):
    resp = await client.get("/api/v1/images/i/nonexistent.png/full")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_nonexistent_thumbnail_returns_404(client):
    resp = await client.get("/api/v1/images/i/nonexistent.png/thumbnail")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_nonexistent_metadata_returns_404(client):
    resp = await client.get("/api/v1/images/i/nonexistent.png/metadata")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_star_nonexistent_image_returns_404(client):
    resp = await client.patch("/api/v1/images/i/nonexistent.png/star?starred=true")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_full_image_served(client, tmp_path, monkeypatch):
    """Create a real image file and serve it via the API."""
    from backend.config import settings
    img_path = settings.images_dir / "test_img.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (64, 64), "red").save(img_path)

    resp = await client.get("/api/v1/images/i/test_img.png/full")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")


@pytest.mark.asyncio
async def test_media_type_matches_extension(client):
    """Legacy .png files and new .jpg files each get the right MIME type."""
    from backend.config import settings
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (64, 64), "red").save(settings.images_dir / "legacy.png")
    PILImage.new("RGB", (64, 64), "red").save(settings.images_dir / "new.jpg")

    resp = await client.get("/api/v1/images/i/legacy.png/full")
    assert resp.headers["content-type"] == "image/png"
    resp = await client.get("/api/v1/images/i/new.jpg/full")
    assert resp.headers["content-type"] == "image/jpeg"

