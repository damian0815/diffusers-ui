"""Integration tests for /api/v1/queue/* endpoints."""
import pytest
from unittest.mock import AsyncMock

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


# ---------------------------------------------------------------------------
# WebSocket event broadcasts on queue mutations
# ---------------------------------------------------------------------------

def _last_broadcast(broadcaster, event_type):
    """Return the most recent broadcast of *event_type* (or None)."""
    calls = [c for c in broadcaster.broadcast_event.await_args_list
             if c.args[0] == event_type]
    return calls[-1] if calls else None


@pytest.fixture
def mock_broadcaster(client):
    """Replace the app broadcaster with a mock and pause the worker so no
    background completion events interfere with assertions (the mock
    pipeline would happily "generate" otherwise)."""
    from unittest.mock import AsyncMock
    worker = client._transport.app.state.worker
    broadcaster = client._transport.app.state.broadcaster
    worker.pause()
    broadcaster.broadcast_event = AsyncMock()
    return broadcaster


@pytest.mark.asyncio
async def test_cancel_pending_item_broadcasts_event(client, mock_broadcaster):
    data = await enqueue_one(client)
    item_id = data["item_ids"][0]
    resp = await client.delete(f"/api/v1/queue/default/i/{item_id}")
    assert resp.status_code == 200
    assert resp.json()["cancelled"] is True
    call = _last_broadcast(mock_broadcaster, "queue_item_cancelled")
    assert call is not None
    assert call.args[1] == {"item_id": item_id}


@pytest.mark.asyncio
async def test_clear_queue_broadcasts_cancelled_events(client, mock_broadcaster):
    first = await enqueue_one(client)
    second = await enqueue_one(client)
    ids = {first["item_ids"][0], second["item_ids"][0]}
    resp = await client.put("/api/v1/queue/default/clear")
    assert resp.status_code == 200
    assert resp.json()["cancelled_count"] == 2
    calls = [c for c in mock_broadcaster.broadcast_event.await_args_list
             if c.args[0] == "queue_item_cancelled"]
    assert {c.args[1]["item_id"] for c in calls} == ids


@pytest.mark.asyncio
async def test_move_front_broadcasts_queue_updated(client, mock_broadcaster):
    await enqueue_one(client)
    second = await enqueue_one(client)
    item_id = second["item_ids"][0]
    resp = await client.put(f"/api/v1/queue/default/i/{item_id}/move_front")
    assert resp.status_code == 200
    call = _last_broadcast(mock_broadcaster, "queue_updated")
    assert call is not None
    assert call.args[1] == {"action": "move_front", "item_id": item_id}


@pytest.mark.asyncio
async def test_prune_broadcasts_queue_updated(client, mock_broadcaster):
    data = await enqueue_one(client)
    item_id = data["item_ids"][0]
    await client.delete(f"/api/v1/queue/default/i/{item_id}")  # make it terminal
    mock_broadcaster.broadcast_event.reset_mock()
    resp = await client.put("/api/v1/queue/default/prune")
    assert resp.status_code == 200
    assert resp.json()["pruned_count"] == 1
    call = _last_broadcast(mock_broadcaster, "queue_updated")
    assert call is not None
    assert call.args[1]["action"] == "prune"


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
async def test_list_is_ordered_by_priority_then_serial(client):
    """Higher rank renders before lower rank; equal ranks keep submission order."""
    await enqueue_one(client)                  # rank 1  (serial 1)
    await enqueue_one(client)                  # rank 1  (serial 2)
    high_body = make_enqueue_body()
    high_body["batch"]["priority"] = 5
    await client.post("/api/v1/queue/default/enqueue_batch", json=high_body)
    mid_body = make_enqueue_body()
    mid_body["batch"]["priority"] = 3
    await client.post("/api/v1/queue/default/enqueue_batch", json=mid_body)

    resp = await client.get("/api/v1/queue/default/")
    ranks = [it["priority_value"] for it in resp.json()["items"]]
    assert ranks == [5, 3, 1, 1]


@pytest.mark.asyncio
async def test_list_queue(client):
    await enqueue_one(client)
    resp = await client.get("/api/v1/queue/default/")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) >= 1


# ---------------------------------------------------------------------------
# Paged / filtered listing (backend-computed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_queue_pagination(client):
    for _ in range(5):
        await enqueue_one(client)

    resp = await client.get("/api/v1/queue/default/?limit=2")
    data = resp.json()
    assert len(data["items"]) == 2
    assert data["total"] == 5
    assert data["has_more"] is True
    first_ids = [i["item_id"] for i in data["items"]]

    resp = await client.get("/api/v1/queue/default/?limit=2&offset=2")
    data = resp.json()
    assert len(data["items"]) == 2
    assert data["has_more"] is True

    resp = await client.get("/api/v1/queue/default/?limit=2&offset=4")
    data = resp.json()
    assert len(data["items"]) == 1
    assert data["has_more"] is False

    # Pages don't overlap.
    all_ids = first_ids + [i["item_id"] for i in (
        (await client.get("/api/v1/queue/default/?limit=2&offset=2")).json()["items"]
    )]
    assert len(set(all_ids)) == 4


@pytest.mark.asyncio
async def test_list_queue_metadata(client):
    await enqueue_one(client)
    data = (await client.get("/api/v1/queue/default/")).json()
    for key in ("summary", "models", "total", "has_more", "priority_count"):
        assert key in data
    assert data["total"] == 1
    assert data["summary"]["pending"] >= 1
    assert "priority" in data["items"][0]
    assert "created_at" in data["items"][0]


@pytest.mark.asyncio
async def test_list_queue_status_filter(client):
    data = await enqueue_one(client)
    item_id = data["item_ids"][0]
    await client.delete(f"/api/v1/queue/default/i/{item_id}")

    resp = await client.get("/api/v1/queue/default/?status=cancelled")
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["status"] == "cancelled"

    resp = await client.get("/api/v1/queue/default/?status=bogus,pending")
    data = resp.json()
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_list_queue_active_state(client):
    data = await enqueue_one(client)
    item_id = data["item_ids"][0]
    await client.delete(f"/api/v1/queue/default/i/{item_id}")

    resp = await client.get("/api/v1/queue/default/?status=pending,processing")
    data = resp.json()
    assert data["total"] == 0  # the only item was cancelled


@pytest.mark.asyncio
async def test_list_queue_model_filter(client):
    await enqueue_one(client, batch=dict(
        prompt="model a prompt", model_key="model_a", board_id="test_board",
    ))
    await enqueue_one(client, batch=dict(
        prompt="model b prompt", model_key="model_b", board_id="test_board",
    ))

    resp = await client.get("/api/v1/queue/default/?model=model_a")
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["model_key"] == "model_a"

    # models list is unfiltered — both models present.
    models = {m["model_key"] for m in data["models"]}
    assert models == {"model_a", "model_b"}


@pytest.mark.asyncio
async def test_list_queue_search_case_insensitive(client):
    await enqueue_one(client, batch=dict(
        prompt="A Beautiful Landscape", model_key="test_model", board_id="test_board",
    ))
    await enqueue_one(client, batch=dict(
        prompt="a grumpy city", model_key="test_model", board_id="test_board",
    ))

    resp = await client.get("/api/v1/queue/default/?q=beautiful")
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["prompt"] == "A Beautiful Landscape"


@pytest.mark.asyncio
async def test_enqueue_prepend_priority_in_list(client):
    data = await enqueue_one(client, prepend=True)
    item_id = data["item_ids"][0]
    resp = await client.get("/api/v1/queue/default/")
    items = resp.json()["items"]
    entry = next(i for i in items if i["item_id"] == item_id)
    assert entry["priority"] is True
    assert resp.json()["priority_count"] >= 1


@pytest.mark.asyncio
async def test_list_sorts_priority_first(client):
    normal = await enqueue_one(client, prepend=False)
    prepended = await enqueue_one(client, prepend=True)

    resp = await client.get("/api/v1/queue/default/")
    items = resp.json()["items"]
    order = [i["item_id"] for i in items]
    prepended_id = prepended["item_ids"][0]
    normal_id = normal["item_ids"][0]
    assert order.index(prepended_id) < order.index(normal_id)
    assert items[order.index(prepended_id)]["priority"] is True


# ---------------------------------------------------------------------------
# VAE slicing (queue-level OOM fallback)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_status_includes_vae_slicing(client):
    resp = await client.get("/api/v1/queue/default/status")
    assert resp.status_code == 200
    assert "vae_slicing" in resp.json()


@pytest.mark.asyncio
async def test_set_vae_slicing(client):
    resp = await client.put(
        "/api/v1/queue/default/vae_slicing", json={"vae_slicing": True}
    )
    assert resp.status_code == 200
    assert resp.json()["vae_slicing"] is True

    status = await client.get("/api/v1/queue/default/status")
    assert status.json()["vae_slicing"] is True

    resp = await client.put(
        "/api/v1/queue/default/vae_slicing", json={"vae_slicing": False}
    )
    assert resp.status_code == 200
    assert resp.json()["vae_slicing"] is False

    status = await client.get("/api/v1/queue/default/status")
    assert status.json()["vae_slicing"] is False


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

