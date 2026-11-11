"""Integration tests for gallery comparison mode (plan COMPARISON_GALLERY_MODE.md §6.2).

Uses the shared `client` fixture: real FastAPI app, SQLite in tmp_path, real
ModelRegistry at tmp_path/models.json. Images are seeded as raw ORM rows (there
is no image-creation API short of running the pipeline).
"""
from __future__ import annotations

import backend.gallery.db as gallery_db
from backend.gallery.compare import comparison_hash

DEFAULT_FIELDS = ["prompt", "negative_prompt", "seed", "width", "height", "cfg"]


async def create_board(client, name: str) -> str:
    resp = await client.post(f"/api/v1/boards/?board_name={name}")
    assert resp.status_code == 200
    return resp.json()["board_id"]


async def install_model(client, name: str) -> str:
    """Register a model via the API; returns its key. Insertion order = added order."""
    resp = await client.post(
        "/api/v2/models/install?source=runwayml/stable-diffusion-v1-5",
        json={"name": name, "base": "sd-1"},
    )
    assert resp.status_code == 200
    return resp.json()["key"]


async def seed_image(board_id: str, image_name: str, **cols) -> None:
    """Insert one ImageRecord on the given board via the patched test session factory."""
    from backend.gallery.models import Gallery, ImageRecord

    async with gallery_db.AsyncSessionLocal() as session:
        gallery_id = (
            await session.execute(
                Gallery.__table__.select().where(Gallery.board_id == board_id)
            )
        ).first().id
        session.add(ImageRecord(image_name=image_name, gallery_id=gallery_id, **cols))
        await session.commit()


async def seed_dataset(client) -> dict:
    """Register A, B and seed the standard fixture rows. Returns ids/keys."""
    key_a = await install_model(client, "Model A")
    key_b = await install_model(client, "Model B")
    board1 = await create_board(client, "Board 1")
    board2 = await create_board(client, "Board 2")

    common = dict(prompt="P", negative_prompt="", width=512, height=512, cfg=7.5, steps=20)
    # A and B each render P/seed 42 (the match pair); B's is the newer submission.
    await seed_image(board1, "img_a42.png", model_key=key_a, model_name="Model A", seed=42,
                     request_serial_number=10, **common)
    await seed_image(board2, "img_b42.png", model_key=key_b, model_name="Model B", seed=42,
                     request_serial_number=20, **common)
    # B also renders P with seed 43 — must never match the 42-pair.
    await seed_image(board1, "img_b43.png", model_key=key_b, model_name="Model B", seed=43,
                     request_serial_number=30, **common)
    # img2img source-board row: empty model_key, identical params otherwise.
    await seed_image(board1, "source_row.png", model_key="", model_name="", seed=42,
                     request_serial_number=40, **common)
    # Row from a model deleted from the registry whose images remain.
    await seed_image(board2, "img_orphan.png", model_key="orphan", model_name="Ghost Model",
                     seed=42, request_serial_number=50, **common)

    return {"key_a": key_a, "key_b": key_b, "board1": board1, "board2": board2}


def compare_params(reference="img_a42.png", fields=DEFAULT_FIELDS, models=(), **kw) -> str:
    """Query string in FastAPI list style: one repeated param per value."""
    qs = f"reference={reference}"
    qs += "".join(f"&fields={f}" for f in fields)
    qs += "".join(f"&models={m}" for m in models)
    qs += "".join(f"&{k}={v}" for k, v in kw.items())
    return qs


async def get_compare(client, **kw):
    return await client.get(f"/api/v1/images/compare?{compare_params(**kw)}")


# ---------------------------------------------------------------------------
# §6.2 cases
# ---------------------------------------------------------------------------

async def test_compare_default_fields_matches_other_model(client):
    ds = await seed_dataset(client)
    resp = await get_compare(client, models=[ds["key_a"], ds["key_b"]])
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    names = {i["image_name"] for i in data["items"]}
    assert names == {"img_a42.png", "img_b42.png"}
    assert data["comparison_hash"]
    assert data["fields"] == DEFAULT_FIELDS
    # Every returned item hashes to the reference hash.
    for item in data["items"]:
        assert comparison_hash(item, DEFAULT_FIELDS) == data["comparison_hash"]


async def test_compare_excludes_nonmatching_seed(client):
    ds = await seed_dataset(client)
    data = (await get_compare(client, models=[ds["key_a"], ds["key_b"]])).json()
    names = {i["image_name"] for i in data["items"]}
    assert "img_b43.png" not in names
    # Source row shares the params but is excluded by the model filter (empty key);
    # the orphan row is not in the registered [A, B] filter either.
    assert "source_row.png" not in names
    assert "img_orphan.png" not in names


async def test_compare_with_model_field(client):
    """Checking `model` puts the model into the identity ⇒ only R's own model matches."""
    ds = await seed_dataset(client)
    data = (await get_compare(
        client, fields=DEFAULT_FIELDS + ["model"], models=[ds["key_a"], ds["key_b"]],
    )).json()
    assert data["total"] == 1
    assert data["items"][0]["image_name"] == "img_a42.png"


async def test_compare_model_filter_restricts(client):
    ds = await seed_dataset(client)
    data = (await get_compare(client, models=[ds["key_a"]])).json()
    assert data["total"] == 1
    assert data["items"][0]["image_name"] == "img_a42.png"


async def test_compare_unknown_field_400(client):
    await seed_dataset(client)
    resp = await get_compare(client, fields=["prompt", "steps"])
    assert resp.status_code == 400
    assert "steps" in resp.json()["detail"]


async def test_compare_missing_reference_404(client):
    await seed_dataset(client)
    resp = await get_compare(client, reference="nope.png")
    assert resp.status_code == 404


async def test_comparison_models_ordering_and_counts(client):
    """Registered in added order, then the removed tail; never-used models still listed."""
    ds = await seed_dataset(client)
    key_c = await install_model(client, "Model C")  # third, never used

    resp = await client.get("/api/v1/images/comparison_models")
    assert resp.status_code == 200
    models = resp.json()["models"]

    assert [m["model_key"] for m in models] == [ds["key_a"], ds["key_b"], key_c, "orphan"]
    by_key = {m["model_key"]: m for m in models}
    assert by_key[key_c]["image_count"] == 0
    assert by_key[key_c]["registered"] is True
    assert by_key[key_c]["last_used"] is None
    assert by_key[ds["key_a"]]["image_count"] == 1
    assert by_key[ds["key_b"]]["image_count"] == 2
    orphan = by_key["orphan"]
    assert orphan["registered"] is False
    assert orphan["image_count"] == 1
    assert orphan["model_name"] == "Ghost Model"  # snapshot from its newest row
    # img2img source rows (empty model_key) are excluded entirely
    assert all(m["model_key"] != "" for m in models)


async def test_compare_pagination(client):
    ds = await seed_dataset(client)
    keys = [ds["key_a"], ds["key_b"]]

    first = (await get_compare(client, models=keys, limit=1)).json()
    assert first["total"] == 2
    assert len(first["items"]) == 1
    # newest submission first (§3.5): B's image has request_serial_number 20
    assert first["items"][0]["image_name"] == "img_b42.png"

    second = (await get_compare(client, models=keys, limit=1, offset=1)).json()
    assert second["total"] == 2
    assert second["items"][0]["image_name"] == "img_a42.png"


async def test_compare_across_boards(client):
    """Matches come from any board and carry their board_id/board_name."""
    ds = await seed_dataset(client)
    data = (await get_compare(client, models=[ds["key_a"], ds["key_b"]])).json()
    assert data["total"] == 2
    by_name = {i["image_name"]: i for i in data["items"]}
    assert by_name["img_a42.png"]["board_id"] == ds["board1"]
    assert by_name["img_b42.png"]["board_id"] == ds["board2"]
    assert by_name["img_a42.png"]["board_name"] == "Board 1"
    assert by_name["img_b42.png"]["board_name"] == "Board 2"
