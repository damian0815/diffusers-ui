"""Integration tests for /api/v1/loras/* endpoints."""
import pytest
import torch
from safetensors.torch import save_file


def _make_lora_folder(root, name):
    """Create root/<name>/pytorch_lora_weights.safetensors with a tiny tensor."""
    from backend.model_registry.scanner import LORA_WEIGHTS_FILENAME
    lora_dir = root / name
    lora_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {"lora_transformer_blocks.0.attn1.to_q.lora_A.weight": torch.zeros(2, 2)},
        str(lora_dir / LORA_WEIGHTS_FILENAME),
    )
    return lora_dir


@pytest.mark.asyncio
async def test_list_loras_empty(client):
    resp = await client.get("/api/v1/loras/")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_install_lora_from_folder(client, tmp_path):
    lora_dir = _make_lora_folder(tmp_path, "my_lora")
    resp = await client.post("/api/v1/loras/install", json={"source": str(lora_dir)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "my_lora"
    assert data["filename"] == "my_lora.safetensors"
    assert data["base_models"] == ["sana"]

    listed = await client.get("/api/v1/loras/")
    names = [e["name"] for e in listed.json()]
    assert "my_lora" in names


@pytest.mark.asyncio
async def test_install_lora_idempotent(client, tmp_path):
    lora_dir = _make_lora_folder(tmp_path, "detail_lora")
    first = await client.post("/api/v1/loras/install", json={"source": str(lora_dir)})
    second = await client.post("/api/v1/loras/install", json={"source": str(lora_dir)})
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["filename"] == second.json()["filename"]


@pytest.mark.asyncio
async def test_install_lora_missing_source_returns_404(client):
    resp = await client.post(
        "/api/v1/loras/install", json={"source": "/does/not/exist"}
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_scan_local_dir_returns_lora_candidates(client, tmp_path):
    _make_lora_folder(tmp_path, "style_lora")
    resp = await client.get(f"/api/v2/models/scan_local_dir?path={tmp_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["candidates"]) == 1
    candidate = data["candidates"][0]
    assert candidate["kind"] == "lora"
    assert candidate["name"] == "style_lora"
