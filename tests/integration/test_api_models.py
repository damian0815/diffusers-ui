"""Integration tests for /api/v2/models/* endpoints."""
import pytest

from tests.conftest import make_fake_sana_dir


async def _install_sana(client, path, name):
    resp = await client.post(
        f"/api/v2/models/install?source={path}",
        json={"name": name, "base": "sana", "source_type": "local_path"},
    )
    assert resp.status_code == 200
    return resp.json()["key"]


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


# ---------------------------------------------------------------------------
# Merge model endpoints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_merge_model(client, tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    d2 = make_fake_sana_dir(tmp_path, "b", ["a", "b"])
    k1 = await _install_sana(client, d1, "Sana A")
    k2 = await _install_sana(client, d2, "Sana B")

    resp = await client.post(
        "/api/v2/models/merge",
        json={"name": "My Blend", "components": [{"key": k1, "weight": 1.0}, {"key": k2, "weight": 2.0}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "My Blend"
    assert data["base"] == "sana"
    assert data["merge"] is not None
    comps = data["merge"]
    assert len(comps) == 2
    assert [c["key"] for c in comps] == [k1, k2]
    assert [c["name"] for c in comps] == ["Sana A", "Sana B"]
    assert [c["weight"] for c in comps] == [1.0, 2.0]

    listing = await client.get("/api/v2/models/")
    merged = next(m for m in listing.json()["models"] if m["key"] == data["key"])
    assert merged["merge"] is not None


@pytest.mark.asyncio
async def test_create_merge_architecture_mismatch(client, tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    d2 = make_fake_sana_dir(tmp_path, "b", ["a", "b", "c"])
    k1 = await _install_sana(client, d1, "Sana A")
    k2 = await _install_sana(client, d2, "Sana B")

    resp = await client.post(
        "/api/v2/models/merge",
        json={"name": "Bad", "components": [{"key": k1}, {"key": k2}]},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["components"] is not None
    assert k2 in detail["components"]
    assert isinstance(detail["components"][k2], str)
    assert k1 not in detail["components"]


@pytest.mark.asyncio
async def test_create_merge_requires_two(client, tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    k1 = await _install_sana(client, d1, "Sana A")
    resp = await client.post(
        "/api/v2/models/merge",
        json={"name": "Solo", "components": [{"key": k1}]},
    )
    assert resp.status_code == 400
    assert "at least 2" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_merge_rejects_missing_key(client, tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    k1 = await _install_sana(client, d1, "Sana A")
    resp = await client.post(
        "/api/v2/models/merge",
        json={"name": "Ghost", "components": [{"key": k1}, {"key": "nope"}]},
    )
    assert resp.status_code == 400
    assert "No model with key" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_merge_rejects_non_sana(client, tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    k1 = await _install_sana(client, d1, "Sana A")
    sd = await client.post(
        "/api/v2/models/install?source=org/sd",
        json={"name": "SD", "base": "sd-1", "source_type": "hf_repo_id"},
    )
    k2 = sd.json()["key"]
    resp = await client.post(
        "/api/v2/models/merge",
        json={"name": "Mixed", "components": [{"key": k1}, {"key": k2}]},
    )
    assert resp.status_code == 400
    assert "only SANA" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_merge_rejects_nested(client, tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    d2 = make_fake_sana_dir(tmp_path, "b", ["a", "b"])
    d3 = make_fake_sana_dir(tmp_path, "c", ["a", "b"])
    k1 = await _install_sana(client, d1, "Sana A")
    k2 = await _install_sana(client, d2, "Sana B")
    k3 = await _install_sana(client, d3, "Sana C")
    merge = await client.post(
        "/api/v2/models/merge",
        json={"name": "Outer", "components": [{"key": k1}, {"key": k2}]},
    )
    merge_key = merge.json()["key"]
    resp = await client.post(
        "/api/v2/models/merge",
        json={"name": "Nested", "components": [{"key": merge_key}, {"key": k3}]},
    )
    assert resp.status_code == 400
    assert "nested" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_patch_quantization_on_merge_rejected(client, tmp_path):
    d1 = make_fake_sana_dir(tmp_path, "a", ["a", "b"])
    d2 = make_fake_sana_dir(tmp_path, "b", ["a", "b"])
    k1 = await _install_sana(client, d1, "Sana A")
    k2 = await _install_sana(client, d2, "Sana B")
    merge = await client.post(
        "/api/v2/models/merge",
        json={"name": "Blend", "components": [{"key": k1}, {"key": k2}]},
    )
    merge_key = merge.json()["key"]
    resp = await client.patch(f"/api/v2/models/{merge_key}", json={"quantization": "int8"})
    assert resp.status_code == 400
    assert "quantization" in resp.json()["detail"]

