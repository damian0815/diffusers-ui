# diffusers-ui API Reference

Base URL: `/api/v1`

- [Galleries (Boards)](#galleries-boards)
- [Images](#images)
- [Queue](#queue)
- [Models](#models)
- [WebSocket](#websocket)

---

## Galleries (Boards)

Prefix: `/api/v1/boards`

### `GET /`

List all galleries (boards).

**Response**
```json
[
  {
    "board_id": "uuid-string",
    "board_name": "My Gallery",
    "created_at": "2025-01-15 10:30:00.123456",
    "image_count": 42
  }
]
```
`image_count` is only present when returned from the list endpoint.

### `POST /?board_name=My+Gallery`

Create a new gallery.

**Query** `board_name` (str, required)

**Response**
```json
{
  "board_id": "uuid-string",
  "board_name": "My Gallery",
  "created_at": "2025-01-15 10:30:00.123456"
}
```

### `PATCH /{board_id}?board_name=New+Name`

Rename a gallery.

**Path** `board_id` — UUID of the gallery  
**Query** `board_name` (str, required)

**Response** Same as `POST /`.

### `DELETE /{board_id}?include_images=false`

Delete a gallery.

**Path** `board_id`  
**Query** `include_images` (bool, default `false`) — also delete all images in the gallery

**Response**
```json
{ "deleted": "board-id", "deleted_images": ["img1.png", "img2.png"] }
```

---

## Images

Prefix: `/api/v1/images`

### `GET /`

Paginated list of images in a gallery. Ordered by `request_serial_number DESC NULLS LAST, created_at DESC`.

**Query**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `board_id` | str | required | gallery UUID |
| `search` | str | — | full-text search on prompt/negative_prompt |
| `sort_by` | str | `request_serial_number` | `request_serial_number` (submission order) or `created_at` (generation order) |
| `sort_dir` | str | `desc` | `asc` or `desc` |
| `starred_first` | bool | `false` | promote starred images to the top of the sort |
| `starred_only` | bool | `false` | only starred images (legacy filter) |
| `offset` | int | `0` | pagination offset |
| `limit` | int | `50` | page size |

**Response**
```json
{
  "total": 200,
  "offset": 0,
  "limit": 50,
  "items": [ /* image record dicts */ ]
}
```

### `GET /names?board_id=uuid`

Return all image names in a gallery (unpaginated).

**Response**
```json
{ "image_names": ["abc.png", "def.png"] }
```

### `POST /move`

Move images between galleries.

**Body**
```json
{
  "image_names": ["img1.png", "img2.png"],
  "source_board_id": "uuid",
  "target_board_id": "uuid"
}
```

**Response**
```json
{ "moved": 2, "target_board_id": "uuid" }
```

### `GET /i/{image_name}/full`

Serve the full-resolution PNG image file.

### `GET /i/{image_name}/thumbnail`

Serve the thumbnail PNG. Falls back to the full image if the thumbnail doesn't exist.

### `GET /i/{image_name}/metadata`

Return generation metadata for an image.

**Response**
```json
{
  "image_name": "abc123.png",
  "prompt": "a cat",
  "negative_prompt": "",
  "seed": 12345,
  "width": 512,
  "height": 512,
  "steps": 20,
  "cfg": 7.5,
  "cfg_scale": 7.5,
  "cfg_rescale_multiplier": 0.0,
  "sampler": "dpmpp_2m",
  "scheduler": "dpmpp_2m",
  "prediction_type": "default",
  "model_key": "model-key",
  "model_name": "Model Name",
  "starred": false,
  "request_serial_number": 42,
  "created_at": "2025-01-15 10:30:00.123456",
  "tags": ["tag1", "tag2"],
  "source_image": null,
  "source_width": null,
  "source_height": null,
  "strength": 0.75,
  "crop_left": 0,
  "crop_top": 0,
  "crop_width": 0,
  "crop_height": 0
}
```

### `GET /i/{image_name}/gallery`

Return the gallery (board) that contains this image.

**Response**
```json
{ "board_id": "uuid", "board_name": "My Gallery" }
```

Both fields are `null` if the image is orphaned.

### `DELETE /i/{image_name}`

Delete an image (record + file + thumbnail).

**Response**
```json
{ "deleted": "abc123.png" }
```

### `PATCH /i/{image_name}/star?starred=true`

Set the starred flag on an image.

**Query** `starred` (bool, required)

**Response** Full image record dict.

### `POST /i/{image_name}/tags/{tag_name}`

Add a tag to an image.

**Response** Full image record dict.

### `DELETE /i/{image_name}/tags/{tag_name}`

Remove a tag from an image.

**Response** Full image record dict.

### `POST /upload-source`

Upload an image file (PNG/JPEG) to use as an img2img source image. Accepts multipart/form-data.

**Body** `file` — image file (multipart upload)

**Response**
```json
{
  "image_name": "abc123.png",
  "width": 1024,
  "height": 768
}
```

---

## Queue

Prefix: `/api/v1/queue/{queue_id}`

`queue_id` is typically `default`.

### `POST /enqueue_batch`

Enqueue one or more generation jobs. Accepts both the native flat format and the legacy InvokeAI graph format (auto-detected by the presence of `batch.graph`).

**Body (native format)**
```json
{
  "prepend": false,
  "batch": {
    "prompt": "a cat",
    "negative_prompt": "",
    "seed": -1,
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg": 7.5,
    "cfg_rescale_multiplier": 0.0,
    "sampler": "dpmpp_2m",
    "prediction_type": "default",
    "model_key": "model-key",
    "board_id": "uuid",
    "count": 1,
    "init_image": null,
    "strength": 0.75,
    "source_width": null,
    "source_height": null,
    "crop_left": 0,
    "crop_top": 0,
    "crop_width": 0,
    "crop_height": 0
  }
}
```

When `count > 1` and `seed >= 0`, successive items auto-increment the seed.  
When `seed == -1`, each item gets an independent random seed.

**Response**
```json
{
  "queue_id": "default",
  "items": [
    {
      "item_id": 1,
      "status": "pending",
      "image_name": null,
      "prompt": "a cat",
      "negative_prompt": "",
      "seed": 12345,
      "width": 512,
      "height": 512,
      "steps": 20,
      "cfg": 7.5,
      "cfg_rescale_multiplier": 0.0,
      "sampler": "dpmpp_2m",
      "prediction_type": "default",
      "model_key": "model-key",
      "model_name": "Model Name",
      "board_id": "uuid",
      "init_image": null,
      "strength": 0.75,
      "source_width": null,
      "source_height": null,
      "crop_left": 0,
      "crop_top": 0,
      "crop_width": 0,
      "crop_height": 0,
      "starred": false,
      "request_serial_number": null,
      "created_at": null,
      "tags": []
    }
  ]
}
```

### `GET /`

List all tracked queue items.

**Response**
```json
{
  "queue_id": "default",
  "items": [
    {
      "item_id": 1,
      "status": "pending",
      "output_image_name": null,
      "error": null,
      "prompt": "a cat",
      "model_key": "model-key"
    }
  ]
}
```

Status values: `pending`, `processing`, `completed`, `failed`, `cancelled`.

### `GET /i/{item_id}`

Get a single queue item.

**Response**
```json
{
  "queue_id": "default",
  "item_id": 1,
  "status": "completed",
  "output_image_name": "abc123.png",
  "error": null
}
```

### `DELETE /i/{item_id}`

Cancel a pending queue item.

**Response**
```json
{ "queue_id": "default", "item_id": 1, "cancelled": true }
```

### `PUT /i/{item_id}/move_front`

Move a pending item to the front of the queue.

**Response**
```json
{ "queue_id": "default", "item_id": 1, "moved": true }
```

Returns 400 if the item is not pending.

### `POST /i/{item_id}/resubmit?prepend=true`

Clone a queue item (any status) and re-enqueue it.

**Query** `prepend` (bool, default `true`)

**Response**
```json
{
  "queue_id": "default",
  "item_id": 42,
  "resubmitted_from": 1
}
```

### `PUT /cancel_current`

Cancel the in-flight generation.

**Response**
```json
{ "queue_id": "default", "cancelled_ids": [1] }
```

### `GET /status`

Return worker status (paused state and max_batch_size setting).

**Response**
```json
{
  "queue_id": "default",
  "paused": false,
  "max_batch_size": 0
}
```

### `PUT /pause`

Pause batch processing after the current generation finishes.

**Response**
```json
{ "queue_id": "default", "paused": true }
```

### `PUT /resume`

Resume batch processing.

**Response**
```json
{ "queue_id": "default", "paused": false }
```

### `PUT /max_batch_size`

Set a user cap on batch size (0 = auto).

**Body**
```json
{ "max_batch_size": 4 }
```

**Response**
```json
{ "queue_id": "default", "max_batch_size": 4 }
```

### `PUT /clear`

Cancel all pending items.

**Response**
```json
{ "queue_id": "default", "cancelled_count": 5 }
```

### `PUT /prune`

Remove completed/failed/cancelled items from the in-memory tracking.

**Response**
```json
{ "queue_id": "default", "pruned_count": 10 }
```

---

## Models

Prefix: `/api/v2/models`

### `GET /`

List all registered models.

**Response**
```json
{
  "models": [
    {
      "key": "model-key",
      "name": "Model Name",
      "type": "sd-1",
      "base": "sd-1",
      "source": "owner/repo-name",
      "source_type": "hf_repo_id",
      "has_api_key": false,
      "quantization": null,
      "vae_tiling": false,
      "use_compel": true,
      "files_missing": false,
      "merge": null
    }
  ]
}
```

### `GET /get_by_attrs?name=...&type=...&base=...`

Find a model by attributes (InvokeAI-compatible). Returns 404 if not found.

**Query** `name` (str), `type` (str), `base` (str)

**Response** Single model entry dict.

### `POST /install?source=owner/repo`

Register a new model.

**Query** `source` (str, required) — HuggingFace repo ID or local path

**Body**
```json
{
  "name": "Model Name",
  "base": "sd-1",
  "source_type": "hf_repo_id",
  "hf_api_key": null,
  "quantization": null,
  "vae_tiling": false,
  "use_compel": true
}
```

`source_type`: `hf_repo_id` (default) or `local_path`  
`base`: `sd-1`, `sd-2`, `sdxl`, `sana`  
`quantization`: `int8`, `int4`, or `null`  
`vae_tiling`: enable VAE tiling on load (lower VRAM)  
`use_compel`: encode prompts via Compel (default `true`; supports weighted syntax and long prompts)

**Response** Single model entry dict.

### `PATCH /{key}`

Update a model's name, quantization and/or prompt-encoding settings.

**Body** (all fields optional)
```json
{ "name": "New Name", "quantization": "int8", "use_compel": false }
```
Send `"quantization": ""` to clear quantization.  
`use_compel`: `true`/`false` — toggles Compel prompt encoding for the model.

**Response** Updated model entry dict.

### `DELETE /{key}`

Remove a model from the registry.

**Response** `{ "deleted": "model-key" }`

### `POST /merge`

Create a **model merge** — a weighted blend of the `transformer` weights of 2+
registered SANA models. Only a *specification* is stored; no merged weight file
is written. The blend is materialised in CPU RAM at inference time when the
merge is loaded.

**Body**
```json
{
  "name": "My SANA Blend",
  "components": [
    { "key": "component-key-1", "weight": 1.0 },
    { "key": "component-key-2", "weight": 2.0 }
  ]
}
```

`weight` defaults to `1.0`. Alphas are normalized over the sum of all weights.

**Validation (definition time)**
- Structural errors (fewer than 2 components, unknown key, non-SANA model,
  nested merge, non-positive weight) → **400** with a plain-string `detail`.
- Architecture mismatch (a component's transformer key set differs from the
  base/first model) → **400** with a structured `detail`:
  ```json
  {
    "detail": {
      "error": "Merge architecture mismatch",
      "components": { "offending-key": "human-readable reason" }
    }
  }
  ```

**Response** A model entry dict with a `merge` field:
```json
"merge": [
  { "key": "component-key-1", "weight": 1.0, "name": "Sana A" }
]
```
`name` is `"(deleted)"` if the component model has been removed from the registry.

Merge models:
- have `quantization` forced to `null` (PATCH to set one → **400**)
- cannot be used as components of another merge
- are listed in `GET /` with a populated `merge` field

### `GET /loaded`

Return the key of the currently loaded model pipeline.

**Response**
```json
{ "loaded_key": "model-key" }
```

`loaded_key` is `null` when no model is loaded.

### `POST /unload_cache`

Unload the current model from memory/VRAM.

Returns **409 Conflict** if a generation is in progress.

**Response**
```json
{ "unloaded": true }
```

### `GET /scan_hf_cache`

Scan the HuggingFace cache directory for installable models (does not modify the registry).

**Response**
```json
{ "candidates": [ /* model entry dicts */ ] }
```

### `GET /scan_local_dir?path=/path/to/models`

Scan a local directory for installable models (does not modify the registry).

**Query** `path` (str, required)

**Response**
```json
{ "candidates": [ /* model entry dicts */ ] }
```

---

## WebSocket

### `WS /ws/preview`

Live latent preview frames during generation.

The server pushes JSON messages:
```json
{
  "type": "preview",
  "item_id": 1,
  "image_b64": "<base64-encoded JPEG>",
  "step": 5,
  "total_steps": 20
}
```

Domain events are wrapped with a `type` and `event` field:
```json
{
  "type": "event",
  "event": "queue_item_completed",
  "data": { "item_id": 1, "image_name": "abc123.png" }
}
```

Clients should send periodic `ping` text messages to keep the connection alive.

---

## WebSocket Broadcast Events

The server broadcasts these events via the shared broadcaster (all connected WebSocket clients):

| Event | Detail | When |
|-------|--------|------|
| `queue_item_enqueued` | full image-record dict | new job enqueued |
| `queue_item_completed` | `{item_id, image_name}` | job finished |
| `queue_item_failed` | `{item_id, error}` | job failed |
| `queue_item_cancelled` | `{item_id}` | job cancelled |
| `worker_paused` | `{}` | worker paused after current batch |
| `worker_resumed` | `{}` | worker resumed |
| `image_created` | full image-record dict | image recorded to gallery |
| `image_deleted` | `{image_name, board_id}` | image removed |
| `images_moved` | `{image_names, source_board_id, target_board_id}` | images moved between galleries |
| `board_created` | `{board_id, board_name, created_at}` | new gallery created |
| `board_renamed` | `{board_id, board_name, created_at}` | gallery renamed |
| `board_deleted` | `{board_id}` | gallery deleted |

These events are delivered to all WebSocket clients connected to `/ws/preview` and are also forwarded to the frontend via `ws-gallery-event` DOM events.
