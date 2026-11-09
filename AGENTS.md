# diffusers-ui — Agent Guidance

## Run

```bash
uv sync
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8765 --reload
# Open http://localhost:8765
```

## Test

```bash
uv run pytest                              # all tests
uv run pytest tests/unit/                  # unit only (no DB needed)
uv run pytest tests/integration/           # integration (fast, uses in-memory SQLite)
uv run pytest tests/integration/test_api_queue.py -k test_cancel  # single test
```

Tests use `pytest-asyncio` with `asyncio_mode = auto` (configured in `pyproject.toml`). The `client` fixture in `tests/conftest.py` wires up a real FastAPI app with mocks — no GPU required. JS in `frontend/` is not linted. Do NOT run `uv run ruff check` — the repo has many pre-existing lint errors that are intentionally left unfixed.

## Architecture

- **Single-page app** — no bundler, no npm. Frontend is Alpine.js (loaded from `frontend/vendor/`) in a single `index.html`. All JS lives in `frontend/*.js` — edit directly, reload to see changes.
- **Backend** — FastAPI with module-level singletons wired in `backend/main.py` via `app.state`. Key singletons: `QueueManager`, `ModelRegistry`, `PipelineLoader`, `PreviewBroadcaster`, `PipelineWorker`.
- **No REST polling for queue updates** — the frontend relies on WebSocket events (`queue_item_completed`, `queue_item_cancelled`, etc.) pushed from `PreviewBroadcaster.broadcast_event()`. The `ws-gallery-event` DOM event bridges WS → Alpine.
- **Alpine component registration** — always use `x-data="componentName()"` with parentheses (direct function call from a global `function` declaration). Do NOT use `Alpine.data()` registration with bare `x-data="componentName"`; the `alpine:init` event can fire before the registration callback runs, causing "not defined" errors.
- **Cross-component state** — the `app()` and `gallery()` Alpine components talk exclusively via `window.dispatchEvent(new CustomEvent(...))`. Don't try to share data directly.

## Key files

| File | Role |
|---|---|
| `backend/main.py` | App factory, lifespan (worker start/stop), singleton wiring |
| `backend/domain.py` | Core models (`QueueItem`, `ImageGenerationParams`, `ItemStatus`, `GenerationCancelled`) |
| `backend/queue/manager.py` | In-memory deque + async SQLite persistence |
| `backend/pipeline/worker.py` | Background asyncio worker — drains queue, runs diffusers pipeline |
| `backend/api/queue.py` | Queue API routes (`/api/v1/queue/default/...`) |
| `backend/api/websocket.py` | WebSocket endpoint + broadcaster forwarding |
| `backend/config.py` | Singleton settings — auto-detects CUDA/MPS/CPU |
| `backend/model_registry/scanner.py` | `scan_hf_cache()` + recursive `scan_local_dir()`; returns scan candidates |
| `backend/api/loras.py` | LoRA discovery + install endpoints |
| `frontend/index.html` | Single-page HTML with Alpine `x-data`, `x-show`, `@click` etc. |
| `frontend/app.js` | Main `app()` Alpine component (~960 lines) |
| `frontend/gallery.js` | Gallery `gallery()` component |
| `frontend/preview.js` | WebSocket client for live preview frames |
| `frontend/style.css` | Design system (~1550 lines) |

## Cancellation

- **Cancel in-flight generation**: `PUT /api/v1/queue/default/cancel_current` — sets a `threading.Event` on the worker, `on_step` callback raises `GenerationCancelled` on the next denoising step.
- Cancel a **pending** item: `DELETE /api/v1/queue/default/i/{item_id}`.
- The frontend cancel button (`.btn-cancel`) is shown when `isGenerating` is true.

## API endpoints

- `POST /api/v1/queue/default/enqueue_batch` — the primary generation endpoint. Accepts native format or legacy InvokeAI graph format (auto-detected by presence of `batch.graph`).
- `GET /api/v1/queue/default/` — list all items with status.
- `PUT /cancel_current` — cancel in-flight generation.
- `PUT /clear` — cancel all pending items.
- `PUT /prune` — remove terminal (completed/failed/cancelled) items from tracking.
- `POST /i/{item_id}/resubmit` — re-enqueue any item.
- LoRA endpoints (`/api/v1/loras/`): `GET /` lists `.safetensors`/`.pt` files in `settings.loras_dir`; `POST /install` copies a weight file (or folder with `pytorch_lora_weights.safetensors`) into `loras_dir`.
- Full API reference in `README_API.md`.

## Notable gotchas

- The `PipelineWorker._run_batch` runs inference in a thread pool via `asyncio.to_thread(_run_pipeline)` — the step callback `on_step` must use thread-safe patterns (e.g. `threading.Event`, not `asyncio.Event`).
- Database uses SQLAlchemy async with `aiosqlite`. Schema migrations are done inline in `backend/gallery/db.py:_apply_inline_migrations()` — there's also an Alembic setup but it's unused for main tables.
- Resolution step is 8 for SD1/SDXL, 32 for SANA. The frontend snaps width/height via `snapResolution()`.
- `CriticalParams` dataclass controls batch grouping — items with different `model_key`, `width`/`height`, `init_image`, etc. go to separate pipeline calls.
- The `_cancel_event` is cleared at the start of each `_run_batch()` so new generations aren't pre-cancelled.

## App lifecycle (startup / shutdown)

Module-level singletons created at import (`backend/main.py`):
`queue_store`, `queue_manager`, `model_registry`, `pipeline_loader`, `broadcaster`, `worker`.

Lifespan startup sequence:
1. Create `images/` and `images/thumbs/` directories
2. `init_db()` — `Base.metadata.create_all` + inline schema migrations
3. Clean orphaned image files not referenced by any `ImageRecord`
4. Ensure "Default" gallery exists; record its `board_id`
5. Ensure "Image To Image Sources" board exists (for img2img source preservation)
6. Restore queue from DB — reset "processing" items to "pending", repopulate deque
7. Load persisted `probe_batch_size` / `max_batch_size_override`
8. Initialize submission serial counter from max `request_serial_number` in DB
9. Start worker as `asyncio.create_task(worker.run())`

Graceful shutdown: `worker.stop()` → wait up to 10 min for idle (`worker.wait_idle()`) → cancel task.

## Model loading / unloading

`PipelineLoader` (`backend/model_registry/loader.py`) holds **one model at a time** (`_loaded_key`).
Load is a no-op if already loaded; otherwise unload first.

Unload sequence (4 steps for max VRAM recovery):
1. Strip accelerate `device_map` hooks
2. Move every sub-module to CPU (releases CUDA allocs)
3. Delete pipeline object + `gc.collect()`
4. `torch.cuda.synchronize()` → `torch.cuda.empty_cache()`

SD1/SDXL loading: `DiffusionPipeline.from_pretrained()` with xformers > SDPA/Flash > standard attention.
SANA loading: float32 base, then bfloat16 cast for text_encoder + transformer. Supports int8/int4 via BitsAndBytes with `device_map="balanced"`.

`ModelRegistry` (`backend/model_registry/registry.py`): reads/writes `models.json` keyed by UUID. Each `ModelEntry` has: key (UUID), name, base (sd-1/sd-2/sdxl/sana), source, source_type, optional hf_api_key, optional quantization, optional `kind` (default `"model"`; scan candidates use `"lora"`).

### Model discovery / scanning (`backend/model_registry/scanner.py`)

`scan_local_dir()` walks **recursively** with `os.walk` and finds two marker types:
- **Pipelines** — a directory containing `model_index.json` (candidate `kind="model"`)
- **LoRA folders** — a directory containing `pytorch_lora_weights.safetensors` (candidate `kind="lora"`)

After a match, `dirs[:] = []` prunes descent so pipeline submodules (`unet/`, `vae/`, …) are never re-scanned. `detect_lora_base()` guesses base from safetensors key prefixes (`lora_transformer_*` → sana, `lora_te1_`/`lora_te2_` → sdxl, `lora_te_text_model` → sd-1), falling back to `_guess_base_from_name()`.

### LoRA add flow

The Generate-panel LoRA picker only sees loose `.safetensors`/`.pt` files directly in `settings.loras_dir`, and `PipelineWorker._load_loras` calls `load_lora_weights(settings.loras_dir, weight_name=filename, adapter_name=filename)` — the adapter name is the full filename **including extension**. So a LoRA found anywhere else must be **copied into `loras_dir`** to be usable: `POST /api/v1/loras/install` (`{source, name}`) accepts a weight file or a folder containing `pytorch_lora_weights.safetensors`, copies it in as `<name>.safetensors` (idempotent), and returns a `LoRAEntry`. On the Models page, `kind === 'lora'` candidates route to this endpoint (via `installLoraCandidate()`), hide the quant select, and `isInstalled()` checks against `loraEntries` by name.

## Gallery system

Three tables in `backend/gallery/models.py`:
- **`galleries`**: `id`, `board_id` (UUID, unique), `name`, `created_at`
- **`images`**: `id`, `image_name` (UUID.png, unique), `gallery_id` (FK, nullable), all generation metadata (prompt, negative_prompt, seed, width, height, steps, cfg, cfg_rescale_multiplier, sampler, prediction_type, model_key, model_name), `request_serial_number`, img2img metadata (source_image, source_width, source_height, strength, crop_*), `starred`, `created_at`
- **`tags`** + **`image_tags`** (many-to-many)

`GalleryService` (`backend/gallery/service.py`):
- `list_images()`: paginated, searchable (prompt/negative_prompt), sortable (`request_serial_number`/`created_at`), starred-first, starred-only filter
- `record_image()`: persists generated image; if img2img, copies source into the img2img source board
- `move_images()`: moves images between boards
- `_delete_image_files()`: removes from both `images/` and `images/thumbs/`

Image file paths: `settings.images_dir / image_name` (full), `settings.thumbnails_dir / image_name` (256×256).

Schema migrations: inline in `backend/gallery/db.py:_apply_inline_migrations()` (checks for missing columns, adds via `ALTER TABLE`). Alembic setup exists but is unused for main tables.

## Frontend architecture

Single HTML file (`frontend/index.html`) with Alpine.js loaded from `frontend/vendor/`. No build step.

### Tab views
| Tab | Description |
|---|---|
| **Generate** | Three-panel: left controls, center image display, right gallery. Prompt, negative prompt, model, seed, aspect ratio, steps, CFG, sampler, prediction type, batch count, img2img |
| **Queue** | Full-page queue list with status, actions (move front, cancel, resubmit), worker controls (pause, batch size, unload, prune), VRAM stats, ETA |
| **Errors** | Scrollable error log with timestamps |
| **Models** | HuggingFace install, scan HF cache / local dir (finds pipelines **and** LoRA folders recursively), installed models with rename/delete/quantization |

### Cross-component events
`app()` and `gallery()` communicate exclusively via `CustomEvent`:
- `select-image`, `board-changed`, `galleries-updated`, `gallery-first-image-changed` (gallery → app)
- `set-source-image`, `image-star-updated`, `request-models-list` (gallery → app)
- `receive-models-list` (app → gallery)
- `ws-gallery-event`, `ws-reconnected` (preview.js → both)

### Display priority (center panel)
1. Active generation + preview frame (live from WebSocket)
2. Selected image from gallery
3. Gallery's first image
4. Blank placeholder

### WebSocket flow (`frontend/preview.js`)
Connects to `ws://host/ws/preview`. Sends `ping` every 20s.
- `type: "preview"` — Base64 JPEG latent frames, coalesced via `requestAnimationFrame`
- `type: "event"` — Domain events dispatched as `ws-gallery-event` DOM events
Auto-reconnects on close with 2s delay. On reconnect dispatches `ws-reconnected`.

### Prompt brace expansion
InvokeAI-style `{a|b|c}` syntax produces cartesian product; each variant enqueued separately.

## Queue system

`QueueManager` (`backend/queue/manager.py`): in-memory `deque[QueueItem]` with `asyncio.Lock` + `_new_item_event` (asyncio.Event) for worker notification. Ordering is driven solely by `priority_value` (descending; FIFO within a rank) — there is no prepend/run-now boolean. Batches never mix priority levels. `move_to_front` bumps the item's rank to max+1.
- `next_pending_batch_window()`: waits for items, optional collection window
- `dequeue_matching()`: scans entire queue for items matching a predicate (batch grouping)

`QueueStore` (`backend/queue/store.py`): SQLAlchemy async persistence to `queue_items` table. `upsert()`, `reset_interrupted()`, `load_all()`.

### API endpoints (queue)
| Method | Path | Purpose |
|---|---|---|
| `POST /api/v1/queue/default/enqueue_batch` | Primary generation endpoint |
| `GET /api/v1/queue/default/` | List all items |
| `DELETE /api/v1/queue/default/i/{item_id}` | Cancel pending |
| `PUT /api/v1/queue/default/cancel_current` | Cancel in-flight |
| `PUT /api/v1/queue/default/clear` | Cancel all pending |
| `PUT /api/v1/queue/default/prune` | Remove terminal items |
| `PUT /api/v1/queue/default/pause` | Pause worker |
| `PUT /api/v1/queue/default/resume` | Resume worker |
| `PUT /api/v1/queue/default/max_batch_size` | Set batch size cap |
| `POST /api/v1/queue/default/i/{item_id}/resubmit` | Clone and re-enqueue |
| `PUT /api/v1/queue/default/i/{item_id}/move_front` | Move to front |
| `GET /api/v1/queue/default/status` | Worker status (paused, batch size, VRAM, images/sec) |

### Board / gallery API (`/api/v1/boards`)
| `GET /` | List all galleries with image counts |
| `POST /` | Create gallery |
| `PATCH /{board_id}` | Rename gallery |
| `DELETE /{board_id}` | Delete gallery (optionally with images) |

### Image API (`/api/v1/images`)
| `GET /` | Paginated image list with search/sort |
| `GET /i/{name}/full` | Serve raw PNG |
| `GET /i/{name}/thumbnail` | Serve 256×256 thumbnail |
| `GET /i/{name}/metadata` | Generation metadata |
| `DELETE /i/{name}` | Delete image |
| `PATCH /i/{name}/star` | Toggle star |
| `POST /move` | Move images between boards |
| `POST /upload-source` | Upload img2img source |

## Preview system (`backend/pipeline/preview.py`)

Two methods:
1. **TAESD** (higher quality) — Tiny autoencoder from `taesd/` submodule. Variants: `sd1`, `sdxl`, `sana`. Lazy-loaded, thread-safe.
2. **Approx** (fallback, faster) — Linear RGB projection from latent space using fixed color factor matrices.

`PreviewBroadcaster` maintains WebSocket connection set. Frames are **dropped** for slow consumers; domain events are **never** dropped.


<!-- BEGIN BEADS INTEGRATION v:1 profile:full hash:bacef91e -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Dolt-powered version control with native sync
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update <id> --claim --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task atomically**: `bd update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Quality
- Use `--acceptance` and `--design` fields when creating issues
- Use `--validate` to check description completeness

### Lifecycle
- `bd defer <id>` / `bd supersede <id>` for issue management
- `bd stale` / `bd orphans` / `bd lint` for hygiene
- `bd human <id>` to flag for human decisions
- `bd formula list` / `bd mol pour <name>` for structured workflows

### Sync

bd stores issue history in Dolt:

- Each write auto-commits to Dolt history
- Use `bd dolt push`/`bd dolt pull` for remote sync
- Do not treat `.beads/issues.jsonl` as the sync protocol

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and https://github.com/gastownhall/beads/blob/main/docs/getting-started/quickstart.md.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.

<!-- END BEADS INTEGRATION -->

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->
