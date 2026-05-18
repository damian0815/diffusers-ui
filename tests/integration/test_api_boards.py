"""Integration tests for /api/v1/boards/* endpoints."""
import pytest


async def create_board(client, name="Test Board") -> dict:
    resp = await client.post(f"/api/v1/boards/?board_name={name}")
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.asyncio
async def test_list_boards_returns_list(client):
    resp = await client.get("/api/v1/boards/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_create_board(client):
    data = await create_board(client, "My Gallery")
    assert data["name"] == "My Gallery"
    assert "board_id" in data


@pytest.mark.asyncio
async def test_created_board_appears_in_list(client):
    await create_board(client, "Special Gallery")
    resp = await client.get("/api/v1/boards/")
    names = [g["name"] for g in resp.json()]
    assert "Special Gallery" in names


@pytest.mark.asyncio
async def test_rename_board(client):
    board = await create_board(client, "Old Name")
    resp = await client.patch(
        f"/api/v1/boards/{board['board_id']}?board_name=New+Name"
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "New Name"


@pytest.mark.asyncio
async def test_delete_board(client):
    board = await create_board(client, "To Delete")
    resp = await client.delete(f"/api/v1/boards/{board['board_id']}")
    assert resp.status_code == 200

    list_resp = await client.get("/api/v1/boards/")
    ids = [g["board_id"] for g in list_resp.json()]
    assert board["board_id"] not in ids


@pytest.mark.asyncio
async def test_delete_nonexistent_board_returns_404(client):
    resp = await client.delete("/api/v1/boards/does-not-exist")
    assert resp.status_code == 404

