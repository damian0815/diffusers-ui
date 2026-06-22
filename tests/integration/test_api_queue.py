"""Integration tests for /api/v1/queue/* endpoints."""
import pytest
from tests.conftest import make_enqueue_body

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def enqueue_one(client, **overrides) -> dict:
    resp = await client.post(
        "/api/v1/queue/default/enqueue_batch",
        json=make_enqueue_body(**overrides),
    )
    assert resp.status_code == 200
    data = resp.json()
    # Normalise: API returns {"items": [...]}, tests use item_ids
    if "items" in data and "item_ids" not in data:
        data["item_ids"] = [it["item_id"] for it in data["items"]]
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enqueue_returns_item_id(client):
    data = await enqueue_one(client)
    assert "item_ids" in data
    assert len(data["item_ids"]) == 1
    assert isinstance(data["item_ids"][0], int)


@pytest.mark.asyncio
async def test_enqueue_returns_queue_id(client):
    data = await enqueue_one(client)
    assert data["queue_id"] == "default"


@pytest.mark.asyncio
async def test_get_queue_item_status(client):
    data = await enqueue_one(client)
    item_id = data["item_ids"][0]
    resp = await client.get(f"/api/v1/queue/default/i/{item_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("pending", "processing", "completed", "failed")


@pytest.mark.asyncio
async def test_get_nonexistent_item_returns_404(client):
    resp = await client.get("/api/v1/queue/default/i/99999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_pending_item(client):
    data = await enqueue_one(client)
    item_id = data["item_ids"][0]
    resp = await client.delete(f"/api/v1/queue/default/i/{item_id}")
    assert resp.status_code == 200
    result = resp.json()
    assert result["item_id"] == item_id


@pytest.mark.asyncio
async def test_clear_queue(client):
    await enqueue_one(client)
    await enqueue_one(client)
    resp = await client.put("/api/v1/queue/default/clear")
    assert resp.status_code == 200
    assert "cancelled_count" in resp.json()


@pytest.mark.asyncio
async def test_prune_queue(client):
    resp = await client.put("/api/v1/queue/default/prune")
    assert resp.status_code == 200
    assert "pruned_count" in resp.json()


@pytest.mark.asyncio
async def test_enqueue_prepend(client):
    data = await enqueue_one(client, prepend=True)
    assert len(data["item_ids"]) == 1


@pytest.mark.asyncio
async def test_list_queue(client):
    await enqueue_one(client)
    resp = await client.get("/api/v1/queue/default/")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) >= 1


# ---------------------------------------------------------------------------
# Crop params tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enqueue_with_crop_params(client):
    """Enqueue with crop params — verify 200 response."""
    data = await enqueue_one(client, batch=dict(
        prompt="crop test",
        model_key="test_model",
        board_id="test_board",
        init_image="dummy.png",
        strength=0.8,
        crop_left=0.1,
        crop_top=0.2,
        crop_width=0.5,
        crop_height=0.6,
    ))
    assert len(data["item_ids"]) == 1
    assert data["items"][0]["crop_left"] == 0.1
    assert data["items"][0]["crop_top"] == 0.2
    assert data["items"][0]["crop_width"] == 0.5
    assert data["items"][0]["crop_height"] == 0.6


@pytest.mark.asyncio
async def test_enqueue_without_crop_params_defaults(client):
    """Enqueue without crop params — verify default 0.0 values in queue item."""
    data = await enqueue_one(client, batch=dict(
        prompt="no crop",
        model_key="test_model",
        board_id="test_board",
    ))
    assert data["items"][0]["crop_left"] == 0.0
    assert data["items"][0]["crop_top"] == 0.0
    assert data["items"][0]["crop_width"] == 0.0
    assert data["items"][0]["crop_height"] == 0.0


@pytest.mark.asyncio
async def test_img2img_crop_backward_compat(client):
    """Enqueue with init_image but no crop params — still works (all 0 defaults)."""
    import PIL.Image as PILImage
    from backend.config import settings
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    src_img = PILImage.new("RGB", (256, 256), "blue")
    src_name = "backward_compat.png"
    src_img.save(settings.images_dir / src_name)

    data = await enqueue_one(client, batch=dict(
        prompt="backward compat",
        model_key="test_model",
        board_id="test_board",
        init_image=src_name,
        strength=0.75,
    ))
    assert len(data["item_ids"]) == 1
    assert data["items"][0]["crop_width"] == 0.0

