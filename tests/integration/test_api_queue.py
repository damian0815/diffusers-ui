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
    return resp.json()


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
    body = make_enqueue_body()
    body["prepend"] = True
    resp = await client.post("/api/v1/queue/default/enqueue_batch", json=body)
    assert resp.status_code == 200
    assert len(resp.json()["item_ids"]) == 1


@pytest.mark.asyncio
async def test_list_queue(client):
    await enqueue_one(client)
    resp = await client.get("/api/v1/queue/default/")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) >= 1

