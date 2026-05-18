"""Integration tests for /api/v2/models/* endpoints."""
import pytest


@pytest.mark.asyncio
async def test_list_models_empty(client):
    resp = await client.get("/api/v2/models/")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert isinstance(data["models"], list)


@pytest.mark.asyncio
async def test_install_model(client):
    resp = await client.post(
        "/api/v2/models/install?source=test/model",
        json={"name": "Test Model", "base": "sd-1", "source_type": "hf_repo_id"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Test Model"
    assert "key" in data


@pytest.mark.asyncio
async def test_installed_model_appears_in_list(client):
    await client.post(
        "/api/v2/models/install?source=org/my-model",
        json={"name": "My Model", "base": "sdxl", "source_type": "hf_repo_id"},
    )
    resp = await client.get("/api/v2/models/")
    names = [m["name"] for m in resp.json()["models"]]
    assert "My Model" in names


@pytest.mark.asyncio
async def test_rename_model(client):
    install = await client.post(
        "/api/v2/models/install?source=org/rename-me",
        json={"name": "Old Name", "base": "sd-1", "source_type": "hf_repo_id"},
    )
    key = install.json()["key"]
    resp = await client.patch(f"/api/v2/models/{key}", json={"name": "New Name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New Name"


@pytest.mark.asyncio
async def test_rename_nonexistent_model_returns_404(client):
    resp = await client.patch("/api/v2/models/no-such-key", json={"name": "X"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_model(client):
    install = await client.post(
        "/api/v2/models/install?source=org/to-delete",
        json={"name": "Delete Me", "base": "sd-2", "source_type": "hf_repo_id"},
    )
    key = install.json()["key"]
    resp = await client.delete(f"/api/v2/models/{key}")
    assert resp.status_code == 200

    list_resp = await client.get("/api/v2/models/")
    keys = [m["key"] for m in list_resp.json()["models"]]
    assert key not in keys


@pytest.mark.asyncio
async def test_delete_nonexistent_model_returns_404(client):
    resp = await client.delete("/api/v2/models/no-such-key")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_scan_hf_cache_endpoint(client):
    resp = await client.get("/api/v2/models/scan_hf_cache")
    assert resp.status_code == 200
    assert "candidates" in resp.json()


@pytest.mark.asyncio
async def test_scan_local_dir_endpoint(client, tmp_path):
    # Create a fake pipeline directory
    model_dir = tmp_path / "my_model"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}")

    resp = await client.get(f"/api/v2/models/scan_local_dir?path={tmp_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert "candidates" in data
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["name"] == "my_model"


@pytest.mark.asyncio
async def test_get_by_attrs_not_found(client):
    resp = await client.get("/api/v2/models/get_by_attrs?name=X&type=main&base=sd-1")
    assert resp.status_code == 404

