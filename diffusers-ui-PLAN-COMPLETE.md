# diffusers-ui — Implementation Plan

A self-contained, single-process image generation UI wrapping Hugging Face Diffusers,
with an InvokeAI-compatible API subset, intelligent batching, live latent preview,
and a multi-gallery browser.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Tech Stack](#2-tech-stack)
3. [Project Structure](#3-project-structure)
4. [Clean Code Ground Rules](#4-clean-code-ground-rules)
5. [Phase 0 — Project Scaffold](#phase-0--project-scaffold)
6. [Phase 1 — Queue Manager](#phase-1--queue-manager)
7. [Phase 2 — Model Registry](#phase-2--model-registry)
8. [Phase 3 — Pipeline Worker & Batcher](#phase-3--pipeline-worker--batcher)
9. [Phase 4 — Gallery & Database](#phase-4--gallery--database)
10. [Phase 5 — API Layer](#phase-5--api-layer)
11. [Phase 6 — Frontend](#phase-6--frontend)
12. [Phase 7 — Integration & Polish](#phase-7--integration--polish)
13. [Testing Strategy](#13-testing-strategy)
14. [Running the App](#14-running-the-app)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Single Python Process                  │
│                                                         │
│  ┌──────────┐   REST/WS   ┌─────────────────────────┐  │
│  │ Browser  │ ◄─────────► │     FastAPI  (main.py)   │  │
│  │ (static  │             │                         │  │
│  │  files)  │             │  /api/v1/queue/*        │  │
│  └──────────┘             │  /api/v1/images/*       │  │
│                           │  /api/v1/boards/*       │  │
│                           │  /api/v2/models/*       │  │
│                           │  /ws/preview            │  │
│                           └────────────┬────────────┘  │
│                                        │               │
│          ┌─────────────────────────────┤               │
│          │                             │               │
│   ┌──────▼──────┐             ┌────────▼───────┐       │
│   │   Queue     │             │    Gallery     │       │
│   │  Manager   │             │   (SQLite)     │       │
│   └──────┬──────┘             └────────────────┘       │
│          │                                             │
│   ┌──────▼──────┐    ┌──────────────┐                 │
│   │   Batcher   │    │    Model     │                 │
│   └──────┬──────┘    │   Registry  │                 │
│          │           │  (JSON +    │                 │
│   ┌──────▼──────┐    │  HF cache)  │                 │
│   │  Pipeline   │◄───┴──────────────┘                 │
│   │  Worker    │                                     │
│   │  (asyncio) │  ──► WebSocket preview broadcast    │
│   └─────────────┘  ──► saves image + writes DB       │
└─────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Single port, no build step.** FastAPI serves the frontend as static files from `frontend/`.
  Alpine.js is loaded from a local vendor file (no CDN dependency at runtime).
- **One asyncio event loop.** The pipeline worker runs as an asyncio background task inside
  the same process. No Celery, no Redis, no second process.
- **InvokeAI API subset.** The API surface is limited to what `InvokeInvoker` actually calls,
  plus cancellation and model management. New clients can use `InvokeInvoker` unchanged.
- **Device-agnostic.** Device selection (CUDA / MPS / CPU) is auto-detected at startup and
  exposed in config. No device logic leaks into business code.

---

## 2. Tech Stack

| Concern | Choice | Reason |
|---|---|---|
| Web framework | FastAPI | Async-native, WebSocket support, automatic OpenAPI docs |
| Frontend reactivity | Alpine.js v3 (vendored) | No build step, reactive bindings in HTML attributes |
| Database | SQLite via SQLAlchemy (async) | Zero-setup, file-based, good enough for local use |
| DB migrations | Alembic | Safe schema evolution without dropping data |
| Image generation | Diffusers ≥ 0.26 | `callback_on_step_end` for live preview |
| Preview encoding | Pillow | Encode latent approximation to JPEG for WebSocket push |
| HF cache scanning | `huggingface_hub` | `scan_cache_dir()` API |
| Testing | pytest + pytest-asyncio + httpx | Async test support, in-process HTTP client |
| Type checking | mypy (optional but encouraged) | Catches contract mismatches early |

---

## 3. Project Structure

```
diffusers-ui/
├── backend/
│   ├── main.py                  # App factory, lifespan, router registration
│   ├── config.py                # Settings dataclass; device auto-detection
│   ├── domain.py                # Shared dataclasses: ImageGenerationParams, QueueItem, etc.
│   │
│   ├── queue/
│   │   ├── __init__.py
│   │   └── manager.py           # QueueManager: enqueue, prepend, cancel, status
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── worker.py            # Background task: drain queue → batcher → pipeline
│   │   ├── batcher.py           # Group QueueItems by critical params for batch inference
│   │   └── preview.py           # Latent → approximate RGB; WebSocket broadcast helper
│   │
│   ├── model_registry/
│   │   ├── __init__.py
│   │   ├── registry.py          # JSON sidecar read/write; add, rename, delete entries
│   │   ├── scanner.py           # Scan HF cache + optional local dir → candidate list
│   │   └── loader.py            # Load/unload diffusers pipeline; device placement
│   │
│   ├── gallery/
│   │   ├── __init__.py
│   │   ├── db.py                # SQLAlchemy engine, session factory, Base
│   │   ├── models.py            # ORM: Gallery, Image, Tag, Star
│   │   └── service.py           # GalleryService: CRUD, search, star, tag, paginate
│   │
│   └── api/
│       ├── __init__.py
│       ├── queue.py             # Router: /api/v1/queue/*
│       ├── images.py            # Router: /api/v1/images/*
│       ├── boards.py            # Router: /api/v1/boards/*
│       ├── models.py            # Router: /api/v2/models/*
│       └── websocket.py         # Router: /ws/preview  (live latent stream)
│
├── frontend/
│   ├── index.html               # Shell: 3-panel CSS grid layout
│   ├── vendor/
│   │   └── alpine.min.js        # Vendored Alpine.js v3 (no CDN)
│   ├── app.js                   # Control panel state (Alpine component)
│   ├── gallery.js               # Gallery panel: infinite scroll, search, thumbnails
│   └── preview.js               # WebSocket client: live latent preview
│
├── tests/
│   ├── conftest.py              # Shared fixtures: test DB, mock pipeline, test client
│   ├── unit/
│   │   ├── test_queue_manager.py
│   │   ├── test_batcher.py
│   │   ├── test_preview.py
│   │   ├── test_model_registry.py
│   │   ├── test_scanner.py
│   │   └── test_gallery_service.py
│   └── integration/
│       ├── test_api_queue.py
│       ├── test_api_images.py
│       ├── test_api_boards.py
│       └── test_api_models.py
│
├── models.json                  # Model registry sidecar (created on first run)
├── alembic/                     # Auto-generated by `alembic init`
├── alembic.ini
├── requirements.txt
├── requirements-dev.txt
└── PLAN.md                      # This file
```

**One rule per directory:** if you find yourself importing from a sibling module's internals,
extract a shared type into `domain.py` instead.

---

## 4. Clean Code Ground Rules

Read these before writing any code. They apply everywhere.

### Single Responsibility Principle (SRP)
Each module has **one reason to change**. Examples of correct splits:
- `batcher.py` knows how to group items — it does NOT call the pipeline
- `worker.py` knows when to call the batcher and pipeline — it does NOT know about HTTP
- `registry.py` reads/writes `models.json` — it does NOT scan the filesystem

### Don't Repeat Yourself (DRY)
- All shared types live in `domain.py`. Never define `ImageGenerationParams` twice.
- Database session creation lives only in `gallery/db.py`, injected via FastAPI `Depends`.
- Device detection lives only in `config.py`.

### Naming
- Functions are verbs: `enqueue_item`, `cancel_item`, `load_pipeline`
- Classes are nouns: `QueueManager`, `GalleryService`, `ModelRegistry`
- Booleans are questions: `is_cancelled`, `has_preview`, `use_cpu`
- No abbreviations except universally understood ones (`cfg`, `vae`, `hf`)

### Small Functions
If a function is longer than ~30 lines, ask whether it has more than one responsibility.
Split it. Name the pieces.

### No Magic Numbers
```python
# Bad
sleep(0.1)
latents * 0.18215

# Good
QUEUE_POLL_INTERVAL_S = 0.1
VAE_SCALING_FACTOR = 0.18215  # Standard SD/SDXL latent scaling constant
```
Put constants at the top of the module that owns them, or in `config.py` if shared.

### Error Handling
- Raise specific exceptions with clear messages: `raise ModelNotFoundError(f"No model with key {key}")`
- Define custom exceptions in the module that raises them
- API routes catch domain exceptions and return appropriate HTTP status codes

---

## Phase 0 — Project Scaffold

**Goal:** A runnable skeleton where every subsequent phase slots in cleanly.

### 0.1 Dependencies

**`requirements.txt`**
```
fastapi>=0.111
uvicorn[standard]>=0.29
diffusers>=0.26
transformers>=4.40
accelerate>=0.30
torch>=2.2            # installed separately per platform
Pillow>=10.0
sqlalchemy[asyncio]>=2.0
aiosqlite>=0.20
alembic>=1.13
huggingface_hub>=0.22
pydantic>=2.0
pydantic-settings>=2.0
tqdm
```

**`requirements-dev.txt`**
```
pytest>=8.0
pytest-asyncio>=0.23
httpx>=0.27
anyio>=4.0
```

### 0.2 Config

**`backend/config.py`**

```python
from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    images_dir: Path = Path("images")
    thumbnails_dir: Path = Path("images/thumbs")
    models_json: Path = Path("models.json")
    db_url: str = "sqlite+aiosqlite:///gallery.db"
    thumbnail_size: tuple[int, int] = (256, 256)
    batch_window_s: float = 0.3   # how long to wait collecting items before batching
    preview_every_n_steps: int = 5
    device: str = field(default_factory=_detect_device)

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Singleton — import this everywhere
settings = Settings()
```

### 0.3 Domain Types

**`backend/domain.py`**

Define every shared type here. Import from this module everywhere else.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import uuid

Sampler = Literal[
    'dpmpp_sde_k', 'dpmpp_sde', 'dpmpp_2m_sde_k', 'dpmpp_2m_sde',
    'dpmpp_2m', 'dpmpp_3m', 'euler', 'euler_a', 'ddim', 'lms', 'kdpm_2_a'
]

@dataclass
class ImageGenerationParams:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    cfg_rescale_multiplier: float
    sampler: Sampler
    model_key: str               # key into models.json

@dataclass
class CriticalParams:
    """Params that must match for items to be batched together."""
    width: int
    height: int
    steps: int
    cfg: float
    cfg_rescale_multiplier: float
    sampler: str
    model_key: str

class ItemStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class QueueItem:
    params: ImageGenerationParams
    item_id: int = field(default_factory=lambda: _next_id())
    status: ItemStatus = ItemStatus.PENDING
    output_image_name: str | None = None
    error: str | None = None

# Simple incrementing ID (sufficient for a local app)
_counter = 0
def _next_id() -> int:
    global _counter
    _counter += 1
    return _counter
```

### 0.4 App Entry Point (skeleton)

**`backend/main.py`**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase 3: start pipeline worker here
    yield
    # Phase 3: shut down worker here

app = FastAPI(title="diffusers-ui", lifespan=lifespan)

# Routers registered in Phase 5
# app.include_router(queue_router)
# ...

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

Run with: `uvicorn backend.main:app --host 127.0.0.1 --port 8765 --reload`

### 0.5 Tests for Phase 0

```python
# tests/unit/test_domain.py
from backend.domain import ImageGenerationParams, QueueItem, ItemStatus, CriticalParams

def test_queue_item_gets_unique_ids():
    p = make_params()
    a = QueueItem(params=p)
    b = QueueItem(params=p)
    assert a.item_id != b.item_id

def test_queue_item_defaults_to_pending():
    item = QueueItem(params=make_params())
    assert item.status == ItemStatus.PENDING
```

---

## Phase 1 — Queue Manager

**Goal:** A thread-safe, asyncio-native queue that supports prepend, per-item status
tracking, individual cancellation, and full queue clear.

### 1.1 Implementation

**`backend/queue/manager.py`**

```python
import asyncio
from collections import deque
from backend.domain import QueueItem, ItemStatus

class QueueManager:
    """
    Manages a queue of QueueItems.

    Responsibilities:
    - Accepting new items (append or prepend)
    - Providing the next pending item to the worker
    - Tracking status of all items (pending → processing → completed/failed/cancelled)
    - Cancellation of individual items or the entire queue
    """

    def __init__(self):
        self._queue: deque[QueueItem] = deque()
        self._all_items: dict[int, QueueItem] = {}
        self._lock = asyncio.Lock()
        self._new_item_event = asyncio.Event()

    async def enqueue(self, item: QueueItem, prepend: bool = False) -> QueueItem:
        async with self._lock:
            if prepend:
                self._queue.appendleft(item)
            else:
                self._queue.append(item)
            self._all_items[item.item_id] = item
        self._new_item_event.set()
        return item

    async def next_pending_batch_window(self, window_s: float) -> list[QueueItem]:
        """
        Wait until at least one item is available, then collect all items
        that arrive within `window_s` seconds. Returns the collected items,
        removing them from the queue. Items cancelled during the window are excluded.
        """
        ...

    async def mark_processing(self, item: QueueItem) -> None: ...
    async def mark_completed(self, item: QueueItem, image_name: str) -> None: ...
    async def mark_failed(self, item: QueueItem, error: str) -> None: ...

    async def cancel_item(self, item_id: int) -> bool:
        """Cancel a pending item. Returns False if item is already processing."""
        ...

    async def cancel_all(self) -> int:
        """Cancel all pending items. Returns count cancelled."""
        ...

    async def prune(self) -> int:
        """Remove completed/failed/cancelled items from tracking. Returns count pruned."""
        ...

    def get_item(self, item_id: int) -> QueueItem | None:
        return self._all_items.get(item_id)

    def all_items(self) -> list[QueueItem]:
        return list(self._all_items.values())
```

### 1.2 Tests

**`tests/unit/test_queue_manager.py`**

Cover every public method:

```python
import pytest, asyncio
from backend.queue.manager import QueueManager
from backend.domain import QueueItem, ItemStatus, make_params  # add make_params helper to domain

@pytest.mark.asyncio
async def test_enqueue_append_order():
    qm = QueueManager()
    a = QueueItem(params=make_params())
    b = QueueItem(params=make_params())
    await qm.enqueue(a)
    await qm.enqueue(b)
    items = await qm.next_pending_batch_window(0)
    assert items[0].item_id == a.item_id

@pytest.mark.asyncio
async def test_enqueue_prepend_goes_to_front():
    qm = QueueManager()
    a = await qm.enqueue(QueueItem(params=make_params()))
    b = await qm.enqueue(QueueItem(params=make_params()), prepend=True)
    items = await qm.next_pending_batch_window(0)
    assert items[0].item_id == b.item_id

@pytest.mark.asyncio
async def test_cancel_pending_item():
    qm = QueueManager()
    item = await qm.enqueue(QueueItem(params=make_params()))
    result = await qm.cancel_item(item.item_id)
    assert result is True
    assert qm.get_item(item.item_id).status == ItemStatus.CANCELLED

@pytest.mark.asyncio
async def test_cancel_processing_item_returns_false():
    qm = QueueManager()
    item = await qm.enqueue(QueueItem(params=make_params()))
    await qm.mark_processing(item)
    result = await qm.cancel_item(item.item_id)
    assert result is False

@pytest.mark.asyncio
async def test_cancel_all_only_affects_pending():
    qm = QueueManager()
    pending = await qm.enqueue(QueueItem(params=make_params()))
    processing = await qm.enqueue(QueueItem(params=make_params()))
    await qm.mark_processing(processing)
    count = await qm.cancel_all()
    assert count == 1
    assert qm.get_item(pending.item_id).status == ItemStatus.CANCELLED
    assert qm.get_item(processing.item_id).status == ItemStatus.PROCESSING

@pytest.mark.asyncio
async def test_cancelled_items_excluded_from_batch_window():
    ...

@pytest.mark.asyncio
async def test_prune_removes_terminal_items():
    ...
```

---

## Phase 2 — Model Registry

**Goal:** A JSON-backed registry of known models, plus scanners for the HF cache and
an optional local directory. The registry is the source of truth; scanners discover
*candidates* that the user then registers.

### 2.1 Domain additions

Add to `backend/domain.py`:

```python
@dataclass
class ModelEntry:
    key: str                     # stable UUID, generated on first registration
    name: str                    # user-facing label
    base: Literal['sd-1', 'sd-2', 'sdxl']
    source: str                  # HF repo ID ("stabilityai/...") or absolute local path
    source_type: Literal['hf_repo_id', 'local_path']
    hf_api_key: str | None = None   # stored only if user provided one
```

### 2.2 Registry

**`backend/model_registry/registry.py`**

```python
import json, uuid
from pathlib import Path
from backend.domain import ModelEntry

class ModelRegistry:
    """
    Reads and writes models.json.
    One entry per registered model. Keys are stable UUIDs.
    """

    def __init__(self, path: Path):
        self._path = path
        self._entries: dict[str, ModelEntry] = {}
        self._load()

    def all(self) -> list[ModelEntry]: ...
    def get(self, key: str) -> ModelEntry | None: ...
    def add(self, entry: ModelEntry) -> ModelEntry: ...
    def rename(self, key: str, new_name: str) -> ModelEntry: ...
    def delete(self, key: str) -> None: ...

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text())
            self._entries = {k: ModelEntry(**v) for k, v in raw.items()}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps({k: vars(v) for k, v in self._entries.items()}, indent=2)
        )
```

### 2.3 Scanner

**`backend/model_registry/scanner.py`**

```python
from pathlib import Path
from huggingface_hub import scan_cache_dir
from backend.domain import ModelEntry

def scan_hf_cache() -> list[ModelEntry]:
    """
    Scan ~/.cache/huggingface/hub and return a ModelEntry candidate
    for each repo that looks like a diffusers pipeline.
    Does NOT write to the registry — just returns candidates.
    """
    ...

def scan_local_dir(directory: Path) -> list[ModelEntry]:
    """
    Walk `directory` looking for subdirectories that contain a
    `model_index.json` (diffusers pipeline marker).
    Returns ModelEntry candidates.
    """
    ...

def _looks_like_pipeline(path: Path) -> bool:
    """Return True if path contains model_index.json."""
    return (path / "model_index.json").exists()
```

### 2.4 Loader

**`backend/model_registry/loader.py`**

```python
from diffusers import DiffusionPipeline
import torch
from backend.config import settings
from backend.domain import ModelEntry

class PipelineLoader:
    """
    Loads and unloads a single DiffusionPipeline.
    Only one model is loaded at a time.
    """

    def __init__(self):
        self._pipeline = None
        self._loaded_key: str | None = None

    def load(self, entry: ModelEntry) -> DiffusionPipeline:
        if self._loaded_key == entry.key:
            return self._pipeline
        self._unload()
        kwargs = {"torch_dtype": _dtype_for_device(settings.device)}
        if entry.hf_api_key:
            kwargs["token"] = entry.hf_api_key
        self._pipeline = DiffusionPipeline.from_pretrained(
            entry.source, **kwargs
        ).to(settings.device)
        self._loaded_key = entry.key
        return self._pipeline

    def _unload(self):
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded_key = None
            # Free GPU memory
            if settings.device == "cuda":
                import torch
                torch.cuda.empty_cache()

    @property
    def loaded_key(self) -> str | None:
        return self._loaded_key

def _dtype_for_device(device: str):
    if device == "cpu":
        return torch.float32
    return torch.float16
```

### 2.5 Tests

**`tests/unit/test_model_registry.py`**

```python
import pytest
from pathlib import Path
from backend.model_registry.registry import ModelRegistry
from backend.domain import ModelEntry

@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "models.json")

def test_add_and_retrieve(registry):
    entry = ModelEntry(key="k1", name="My Model", base="sd-2",
                       source="org/repo", source_type="hf_repo_id")
    registry.add(entry)
    assert registry.get("k1").name == "My Model"

def test_rename(registry):
    entry = ModelEntry(key="k1", name="Old", base="sd-1",
                       source="org/repo", source_type="hf_repo_id")
    registry.add(entry)
    registry.rename("k1", "New")
    assert registry.get("k1").name == "New"

def test_delete(registry):
    entry = ModelEntry(key="k1", name="X", base="sd-1",
                       source="org/repo", source_type="hf_repo_id")
    registry.add(entry)
    registry.delete("k1")
    assert registry.get("k1") is None

def test_persists_across_instances(tmp_path):
    path = tmp_path / "models.json"
    r1 = ModelRegistry(path)
    r1.add(ModelEntry(key="k1", name="Persist", base="sd-2",
                      source="org/repo", source_type="hf_repo_id"))
    r2 = ModelRegistry(path)  # new instance, same file
    assert r2.get("k1").name == "Persist"
```

**`tests/unit/test_scanner.py`**

```python
def test_scan_local_dir_finds_pipeline(tmp_path):
    pipeline_dir = tmp_path / "my_model"
    pipeline_dir.mkdir()
    (pipeline_dir / "model_index.json").write_text("{}")
    results = scan_local_dir(tmp_path)
    assert any(r.source == str(pipeline_dir) for r in results)

def test_scan_local_dir_ignores_non_pipelines(tmp_path):
    (tmp_path / "random_folder").mkdir()
    results = scan_local_dir(tmp_path)
    assert results == []
```

---

## Phase 3 — Pipeline Worker & Batcher

**Goal:** A background asyncio task that continuously drains the queue, groups items
into batches by critical params, runs inference, and broadcasts live previews via
WebSocket.

### 3.1 Batcher

**`backend/pipeline/batcher.py`**

The batcher's job is pure grouping logic — no I/O, no pipeline calls.
This makes it trivially unit-testable.

```python
from backend.domain import ImageGenerationParams, QueueItem, CriticalParams

def get_critical_params(params: ImageGenerationParams) -> CriticalParams:
    """
    Extract the params that must match for two items to be batched together.
    Items with different critical params require separate pipeline calls.
    """
    return CriticalParams(
        width=params.width,
        height=params.height,
        steps=params.steps,
        cfg=params.cfg,
        cfg_rescale_multiplier=params.cfg_rescale_multiplier,
        sampler=params.sampler,
        model_key=params.model_key,
    )

def group_into_batches(items: list[QueueItem]) -> list[list[QueueItem]]:
    """
    Group items by critical params, preserving queue order within each group.
    Returns a list of groups; each group can be sent to the pipeline as one batch.

    Example:
        [A(512x512), B(768x768), C(512x512)] →
        [[A, C], [B]]    (A and C share critical params)
    """
    seen: dict[CriticalParams, list[QueueItem]] = {}
    for item in items:
        key = get_critical_params(item.params)
        seen.setdefault(key, []).append(item)
    return list(seen.values())
```

### 3.2 Preview

**`backend/pipeline/preview.py`**

```python
import io
import torch
from PIL import Image

# Standard SD/SDXL latent scaling constant
VAE_SCALING_FACTOR = 0.18215

def latents_to_preview_image(latents: torch.Tensor) -> bytes:
    """
    Convert a latent tensor to a low-quality preview JPEG without using the VAE.
    Uses a channel-average approximation: fast but blurry. Good enough for progress.

    Args:
        latents: shape (1, C, H, W) — single-item latent from a step callback

    Returns:
        JPEG bytes suitable for WebSocket transmission.
    """
    with torch.no_grad():
        # Bring into roughly [0, 1] range
        preview = (latents[0].float() / VAE_SCALING_FACTOR).mean(dim=0)
        preview = ((preview - preview.min()) / (preview.max() - preview.min() + 1e-8))
        preview_np = (preview.cpu().numpy() * 255).clip(0, 255).astype("uint8")

    img = Image.fromarray(preview_np, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    return buf.getvalue()


class PreviewBroadcaster:
    """
    Maintains a set of active WebSocket connections and broadcasts
    preview frames to all of them.

    This class knows nothing about FastAPI — it just holds connections
    and sends bytes. The WebSocket router manages add/remove.
    """

    def __init__(self):
        self._connections: set = set()

    def add(self, ws) -> None:
        self._connections.add(ws)

    def remove(self, ws) -> None:
        self._connections.discard(ws)

    async def broadcast(self, item_id: int, step: int, total_steps: int,
                        jpeg_bytes: bytes) -> None:
        import json, base64
        payload = json.dumps({
            "item_id": item_id,
            "step": step,
            "total_steps": total_steps,
            "image_b64": base64.b64encode(jpeg_bytes).decode(),
        })
        dead = set()
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._connections -= dead
```

### 3.3 Worker

**`backend/pipeline/worker.py`**

```python
import asyncio
from pathlib import Path
from PIL import Image as PILImage

from backend.config import settings
from backend.domain import QueueItem, ItemStatus
from backend.queue.manager import QueueManager
from backend.pipeline.batcher import group_into_batches
from backend.pipeline.preview import latents_to_preview_image, PreviewBroadcaster
from backend.model_registry.loader import PipelineLoader
from backend.model_registry.registry import ModelRegistry

SAMPLER_MAP = {
    # Map domain sampler names to diffusers scheduler classes
    "dpmpp_2m": "DPMSolverMultistepScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    # ... fill in all samplers
}

class PipelineWorker:
    """
    Runs as a single asyncio background task.
    Drains the queue in batches and calls the diffusers pipeline.
    """

    def __init__(self, queue: QueueManager, registry: ModelRegistry,
                 loader: PipelineLoader, broadcaster: PreviewBroadcaster):
        self._queue = queue
        self._registry = registry
        self._loader = loader
        self._broadcaster = broadcaster
        self._running = False

    async def run(self) -> None:
        self._running = True
        while self._running:
            items = await self._queue.next_pending_batch_window(settings.batch_window_s)
            if not items:
                continue
            for batch in group_into_batches(items):
                await self._run_batch(batch)

    async def stop(self) -> None:
        self._running = False

    async def _run_batch(self, batch: list[QueueItem]) -> None:
        # Load model (no-op if already loaded)
        entry = self._registry.get(batch[0].params.model_key)
        pipeline = await asyncio.to_thread(self._loader.load, entry)

        for item in batch:
            await self._queue.mark_processing(item)

        params = batch[0].params  # all share critical params
        prompts = [item.params.prompt for item in batch]
        neg_prompts = [item.params.negative_prompt for item in batch]
        seeds = [item.params.seed for item in batch]

        step_counter = {"n": 0}

        def on_step(pipe, step, timestep, callback_kwargs):
            step_counter["n"] += 1
            if step_counter["n"] % settings.preview_every_n_steps == 0:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    for i, item in enumerate(batch):
                        preview = latents_to_preview_image(latents[i:i+1])
                        asyncio.run_coroutine_threadsafe(
                            self._broadcaster.broadcast(
                                item.item_id, step, params.steps, preview
                            ),
                            asyncio.get_event_loop()
                        )
            return callback_kwargs

        try:
            import torch
            generators = [torch.Generator(device=settings.device).manual_seed(s)
                          for s in seeds]
            result = await asyncio.to_thread(
                pipeline,
                prompt=prompts,
                negative_prompt=neg_prompts,
                width=params.width,
                height=params.height,
                num_inference_steps=params.steps,
                guidance_scale=params.cfg,
                generator=generators,
                callback_on_step_end=on_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
            for item, pil_image in zip(batch, result.images):
                image_name = await asyncio.to_thread(
                    _save_image, pil_image, item
                )
                await self._queue.mark_completed(item, image_name)
        except Exception as e:
            for item in batch:
                await self._queue.mark_failed(item, str(e))
            raise


def _save_image(pil_image: PILImage.Image, item: QueueItem) -> str:
    """Save image + thumbnail; return the image filename."""
    import uuid
    name = f"{uuid.uuid4().hex}.png"
    out_path = settings.images_dir / name
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    pil_image.save(out_path)

    thumb = pil_image.copy()
    thumb.thumbnail(settings.thumbnail_size)
    settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    thumb.save(settings.thumbnails_dir / name)

    return name
```

### 3.4 Tests

**`tests/unit/test_batcher.py`**

```python
from backend.pipeline.batcher import group_into_batches, get_critical_params
from backend.domain import QueueItem

def make_item(width=512, height=512, cfg=7.5, sampler="euler_a",
              steps=30, model_key="m1"):
    return QueueItem(params=make_params(
        width=width, height=height, cfg=cfg,
        sampler=sampler, steps=steps, model_key=model_key
    ))

def test_same_critical_params_grouped_together():
    a, b = make_item(), make_item()
    groups = group_into_batches([a, b])
    assert len(groups) == 1
    assert len(groups[0]) == 2

def test_different_sizes_separate_groups():
    a = make_item(width=512)
    b = make_item(width=768)
    groups = group_into_batches([a, b])
    assert len(groups) == 2

def test_empty_input():
    assert group_into_batches([]) == []

def test_preserves_order_within_group():
    items = [make_item() for _ in range(5)]
    groups = group_into_batches(items)
    assert groups[0] == items
```

**`tests/unit/test_preview.py`**

```python
import torch
from backend.pipeline.preview import latents_to_preview_image

def test_returns_bytes():
    latents = torch.randn(1, 4, 64, 80)
    result = latents_to_preview_image(latents)
    assert isinstance(result, bytes)
    assert len(result) > 0

def test_output_is_valid_jpeg():
    import io
    from PIL import Image
    latents = torch.randn(1, 4, 64, 80)
    result = latents_to_preview_image(latents)
    img = Image.open(io.BytesIO(result))
    assert img.format == "JPEG"
```

---

## Phase 4 — Gallery & Database

**Goal:** SQLite storage for images, galleries (boards), tags, and stars.
A clean service layer keeps all SQL out of the API routes.

### 4.1 ORM Models

**`backend/gallery/models.py`**

```python
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, ForeignKey, Table
from sqlalchemy.orm import relationship
from backend.gallery.db import Base

image_tags = Table(
    "image_tags", Base.metadata,
    Column("image_id", Integer, ForeignKey("images.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)

class Gallery(Base):
    __tablename__ = "galleries"
    id = Column(Integer, primary_key=True)
    board_id = Column(String, unique=True, index=True)  # UUID string
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    images = relationship("ImageRecord", back_populates="gallery")

class ImageRecord(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    image_name = Column(String, unique=True, index=True)   # filename, e.g. "abc123.png"
    gallery_id = Column(Integer, ForeignKey("galleries.id"))
    gallery = relationship("Gallery", back_populates="images")

    # Generation metadata (denormalised for fast search)
    prompt = Column(String)
    negative_prompt = Column(String)
    seed = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    steps = Column(Integer)
    cfg = Column(Float)
    cfg_rescale_multiplier = Column(Float)
    sampler = Column(String)
    model_key = Column(String)
    model_name = Column(String)

    starred = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    tags = relationship("Tag", secondary=image_tags, back_populates="images")

class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    images = relationship("ImageRecord", secondary=image_tags, back_populates="tags")
```

### 4.2 Database Setup

**`backend/gallery/db.py`**

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from backend.config import settings

class Base(DeclarativeBase):
    pass

engine = create_async_engine(settings.db_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def init_db() -> None:
    """Create all tables. Called once at startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_session() -> AsyncSession:
    """FastAPI dependency. Yields a session and commits/rolls back."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### 4.3 Gallery Service

**`backend/gallery/service.py`**

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from backend.gallery.models import Gallery, ImageRecord, Tag
from backend.domain import QueueItem

class GalleryService:
    """
    All database operations. No SQL in route handlers — use this class.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    # --- Galleries / Boards ---
    async def list_galleries(self) -> list[Gallery]: ...
    async def get_gallery(self, board_id: str) -> Gallery | None: ...
    async def create_gallery(self, name: str) -> Gallery: ...
    async def delete_gallery(self, board_id: str) -> None: ...

    # --- Images ---
    async def record_image(self, item: QueueItem, image_name: str,
                           board_id: str, model_name: str) -> ImageRecord: ...

    async def get_image(self, image_name: str) -> ImageRecord | None: ...

    async def list_images(self, board_id: str, search: str | None = None,
                          starred_only: bool = False,
                          offset: int = 0, limit: int = 50) -> list[ImageRecord]:
        """
        Paginated image list with optional full-text search across
        prompt and negative_prompt columns.
        """
        ...

    async def list_image_names(self, board_id: str) -> list[str]: ...

    # --- Stars & Tags ---
    async def set_starred(self, image_name: str, starred: bool) -> ImageRecord: ...
    async def add_tag(self, image_name: str, tag_name: str) -> ImageRecord: ...
    async def remove_tag(self, image_name: str, tag_name: str) -> ImageRecord: ...
```

### 4.4 Tests

**`tests/unit/test_gallery_service.py`**

```python
import pytest
from backend.gallery.db import Base, init_db, AsyncSessionLocal
from backend.gallery.service import GalleryService
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

@pytest.fixture
async def service():
    # Use in-memory SQLite for tests — fast and isolated
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield GalleryService(session)

@pytest.mark.asyncio
async def test_create_and_list_gallery(service):
    await service.create_gallery("Test Board")
    galleries = await service.list_galleries()
    assert any(g.name == "Test Board" for g in galleries)

@pytest.mark.asyncio
async def test_search_by_prompt(service):
    gallery = await service.create_gallery("Board")
    await service.record_image(make_item(prompt="ocean sunset"), "img1.png", gallery.board_id, "model")
    await service.record_image(make_item(prompt="mountain fog"), "img2.png", gallery.board_id, "model")
    results = await service.list_images(gallery.board_id, search="ocean")
    assert len(results) == 1
    assert results[0].image_name == "img1.png"

@pytest.mark.asyncio
async def test_star_image(service): ...

@pytest.mark.asyncio
async def test_add_and_remove_tag(service): ...

@pytest.mark.asyncio
async def test_pagination(service): ...
```

---

## Phase 5 — API Layer

**Goal:** FastAPI routers that expose the InvokeAI-compatible API subset.
Routes are thin: validate input, call a service, return a response.
No business logic in routes.

### 5.1 Register everything in `main.py`

```python
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.config import settings
from backend.gallery.db import init_db
from backend.queue.manager import QueueManager
from backend.model_registry.registry import ModelRegistry
from backend.model_registry.loader import PipelineLoader
from backend.pipeline.preview import PreviewBroadcaster
from backend.pipeline.worker import PipelineWorker
from backend.api import queue, images, boards, models, websocket

# Singletons shared across requests via app.state
queue_manager = QueueManager()
model_registry = ModelRegistry(settings.models_json)
pipeline_loader = PipelineLoader()
broadcaster = PreviewBroadcaster()
worker = PipelineWorker(queue_manager, model_registry, pipeline_loader, broadcaster)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    task = asyncio.create_task(worker.run())
    yield
    await worker.stop()
    task.cancel()

app = FastAPI(title="diffusers-ui", lifespan=lifespan)
app.state.queue = queue_manager
app.state.registry = model_registry
app.state.broadcaster = broadcaster

app.include_router(queue.router)
app.include_router(images.router)
app.include_router(boards.router)
app.include_router(models.router)
app.include_router(websocket.router)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

### 5.2 Queue Router

**`backend/api/queue.py`**

```python
from fastapi import APIRouter, Request, HTTPException
from backend.domain import ImageGenerationParams, QueueItem, ItemStatus

router = APIRouter(prefix="/api/v1/queue/{queue_id}")

@router.post("/enqueue_batch")
async def enqueue_batch(queue_id: str, body: dict, request: Request):
    """
    InvokeAI-compatible enqueue. Accepts the batch body format used by InvokeInvoker.
    Parses it into ImageGenerationParams and adds to queue.
    Returns {"item_ids": [int, ...]}.
    """
    ...

@router.get("/i/{item_id}")
async def get_queue_item(queue_id: str, item_id: int, request: Request):
    """
    Returns item status and output image name if completed.
    Matches InvokeAI's response shape so InvokeInvoker works unchanged.
    """
    ...

@router.delete("/i/{item_id}")
async def cancel_item(queue_id: str, item_id: int, request: Request): ...

@router.put("/clear")
async def clear_queue(queue_id: str, request: Request): ...

@router.put("/prune")
async def prune_queue(queue_id: str, request: Request): ...
```

### 5.3 Boards Router

**`backend/api/boards.py`**

```python
router = APIRouter(prefix="/api/v1/boards")

@router.get("/")
async def list_boards(session=Depends(get_session)): ...

@router.post("/")
async def create_board(board_name: str, session=Depends(get_session)): ...

@router.delete("/{board_id}")
async def delete_board(board_id: str, include_images: bool = False,
                        session=Depends(get_session)): ...
```

### 5.4 Images Router

**`backend/api/images.py`**

```python
router = APIRouter(prefix="/api/v1/images")

@router.get("/names")
async def list_image_names(board_id: str, session=Depends(get_session)): ...

@router.get("/i/{image_name}/full")
async def get_image_full(image_name: str):
    """Serve the raw image file."""
    ...

@router.get("/i/{image_name}/thumbnail")
async def get_thumbnail(image_name: str): ...

@router.get("/i/{image_name}/metadata")
async def get_image_metadata(image_name: str, session=Depends(get_session)): ...

@router.patch("/i/{image_name}/star")
async def set_star(image_name: str, starred: bool, session=Depends(get_session)): ...

@router.get("/")
async def list_images(board_id: str, search: str | None = None,
                      starred_only: bool = False,
                      offset: int = 0, limit: int = 50,
                      session=Depends(get_session)): ...
```

### 5.5 Models Router

**`backend/api/models.py`**

```python
router = APIRouter(prefix="/api/v2/models")

@router.get("/")
async def list_models(request: Request): ...

@router.get("/get_by_attrs")
async def get_by_attrs(name: str, type: str, base: str, request: Request): ...

@router.post("/install")
async def install_model(source: str, body: dict, request: Request): ...

@router.patch("/{key}")
async def rename_model(key: str, body: dict, request: Request): ...

@router.delete("/{key}")
async def delete_model(key: str, request: Request): ...

@router.get("/scan_hf_cache")
async def scan_hf_cache(request: Request):
    """Return HF cache candidates. Does not modify the registry."""
    ...

@router.get("/scan_local_dir")
async def scan_local_dir(path: str, request: Request):
    """Return local dir candidates. Does not modify the registry."""
    ...
```

### 5.6 WebSocket Router

**`backend/api/websocket.py`**

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

router = APIRouter()

@router.websocket("/ws/preview")
async def preview_ws(websocket: WebSocket, request: Request):
    broadcaster = request.app.state.broadcaster
    await websocket.accept()
    broadcaster.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive; client sends pings
    except WebSocketDisconnect:
        broadcaster.remove(websocket)
```

### 5.7 Integration Tests

**`tests/conftest.py`**

```python
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
```

**`tests/integration/test_api_queue.py`**

```python
@pytest.mark.asyncio
async def test_enqueue_returns_item_id(client):
    body = make_enqueue_body()   # helper that builds the InvokeAI batch body
    resp = await client.post("/api/v1/queue/default/enqueue_batch", json=body)
    assert resp.status_code == 200
    assert "item_ids" in resp.json()
    assert len(resp.json()["item_ids"]) == 1

@pytest.mark.asyncio
async def test_get_queue_item_status(client):
    item_id = (await enqueue_one(client))["item_ids"][0]
    resp = await client.get(f"/api/v1/queue/default/i/{item_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] in ["pending", "processing", "completed"]

@pytest.mark.asyncio
async def test_cancel_pending_item(client): ...

@pytest.mark.asyncio
async def test_clear_queue(client): ...

@pytest.mark.asyncio
async def test_enqueue_prepend(client): ...
```

---

## Phase 6 — Frontend

**Goal:** A zero-build-step frontend: one HTML file plus two JS files.
Alpine.js handles reactivity. All layout is CSS Grid.

### 6.1 Layout Shell

**`frontend/index.html`**

Three-panel CSS grid. The panels are sized `280px 1fr 320px` by default
and can be resized by the user via CSS `resize` on the panels.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>diffusers-ui</title>
  <script src="/vendor/alpine.min.js" defer></script>
  <script src="/app.js" defer></script>
  <script src="/gallery.js" defer></script>
  <script src="/preview.js" defer></script>
  <link rel="stylesheet" href="/style.css">
</head>
<body x-data="app()">
  <div id="layout">
    <!-- LEFT: Controls -->
    <aside id="controls-panel">
      <!-- prompt, negative prompt, seed, width, height, steps, cfg,
           cfg_rescale, sampler, model selector, generate button -->
    </aside>

    <!-- MIDDLE: Image display + queue controls -->
    <main id="image-panel">
      <div id="image-display">
        <!-- Shows selected image or live preview -->
      </div>
      <div id="adopt-bar">
        <!-- Adopt: Prompt | Size | Model | Seed | Steps | All -->
        <!-- keyboard shortcuts shown in tooltips -->
      </div>
    </main>

    <!-- RIGHT: Gallery -->
    <aside id="gallery-panel" x-data="gallery()">
      <div id="gallery-list"><!-- gallery names, search --></div>
      <div id="gallery-contents"><!-- image thumbnails, infinite scroll --></div>
    </aside>
  </div>
</body>
</html>
```

### 6.2 Control Panel

**`frontend/app.js`**

```javascript
function app() {
  return {
    // Form state
    prompt: '',
    negativePrompt: '',
    seed: -1,           // -1 = random
    width: 512,
    height: 512,
    steps: 30,
    cfg: 7.5,
    cfgRescale: 0.7,
    sampler: 'dpmpp_2m',
    modelKey: '',

    // UI state
    selectedImageName: null,
    previewSrc: null,
    models: [],

    async init() {
      await this.loadModels();
      this.setupKeyboardShortcuts();
    },

    async generate(prepend = false) {
      const body = this.buildEnqueueBody(prepend);
      const resp = await fetch('/api/v1/queue/default/enqueue_batch', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      // handle response
    },

    buildEnqueueBody(prepend) {
      // Build the InvokeAI-compatible enqueue_batch body
      return { prepend, /* ... */ };
    },

    adoptSettings(from, what) {
      // what = 'prompt' | 'size' | 'model' | 'seed' | 'steps' | 'all'
      const metadata = from;  // ImageRecord metadata object
      if (what === 'all' || what === 'prompt') {
        this.prompt = metadata.prompt;
        this.negativePrompt = metadata.negative_prompt;
      }
      if (what === 'all' || what === 'size') {
        this.width = metadata.width;
        this.height = metadata.height;
      }
      // ... etc
    },

    setupKeyboardShortcuts() {
      document.addEventListener('keydown', (e) => {
        const meta = e.metaKey || e.ctrlKey;
        if (meta && e.key === 'Enter') {
          e.preventDefault();
          this.generate(e.shiftKey);  // shift = prepend
        }
        // adopt shortcuts: meta+1..6
      });
    },

    async loadModels() {
      const resp = await fetch('/api/v2/models/');
      this.models = (await resp.json()).models;
      if (this.models.length > 0) this.modelKey = this.models[0].key;
    },
  };
}
```

### 6.3 Gallery Panel

**`frontend/gallery.js`**

```javascript
function gallery() {
  return {
    galleries: [],
    selectedBoardId: null,
    images: [],
    search: '',
    starredOnly: false,
    offset: 0,
    limit: 50,
    loading: false,
    hasMore: true,

    async init() {
      await this.loadGalleries();
      this.setupInfiniteScroll();
    },

    async loadGalleries() {
      const resp = await fetch('/api/v1/boards/?all=true');
      this.galleries = (await resp.json());
      if (this.galleries.length > 0) {
        await this.selectGallery(this.galleries[0].board_id);
      }
    },

    async selectGallery(boardId) {
      this.selectedBoardId = boardId;
      this.images = [];
      this.offset = 0;
      this.hasMore = true;
      await this.loadMore();
    },

    async loadMore() {
      if (this.loading || !this.hasMore) return;
      this.loading = true;
      const params = new URLSearchParams({
        board_id: this.selectedBoardId,
        offset: this.offset,
        limit: this.limit,
        ...(this.search && {search: this.search}),
        ...(this.starredOnly && {starred_only: true}),
      });
      const resp = await fetch(`/api/v1/images/?${params}`);
      const batch = await resp.json();
      this.images.push(...batch);
      this.offset += batch.length;
      this.hasMore = batch.length === this.limit;
      this.loading = false;
    },

    setupInfiniteScroll() {
      // IntersectionObserver on a sentinel div at the bottom of the grid
      const sentinel = document.getElementById('gallery-sentinel');
      new IntersectionObserver(entries => {
        if (entries[0].isIntersecting) this.loadMore();
      }).observe(sentinel);
    },

    selectImage(imageName) {
      // Tell parent app component to display this image
      this.$dispatch('select-image', {imageName});
    },
  };
}
```

### 6.4 Live Preview

**`frontend/preview.js`**

```javascript
function connectPreview(onFrame) {
  const ws = new WebSocket(`ws://${location.host}/ws/preview`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onFrame(data);   // {item_id, step, total_steps, image_b64}
  };

  // Keep-alive ping every 20s
  const ping = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send('ping');
  }, 20_000);

  ws.onclose = () => {
    clearInterval(ping);
    // Reconnect after a short delay
    setTimeout(() => connectPreview(onFrame), 2_000);
  };

  return ws;
}
```

In `app.js`, wire it up in `init()`:

```javascript
connectPreview(({image_b64, step, total_steps}) => {
  this.previewSrc = `data:image/jpeg;base64,${image_b64}`;
  this.previewProgress = step / total_steps;
});
```

---

## Phase 7 — Integration & Polish

**Goal:** Everything working end-to-end, error handling, loading states.

### 7.1 Startup checks

In `lifespan`:
- Create `images/` and `images/thumbs/` directories if missing
- Create default gallery "Default" if no galleries exist
- Log detected device

### 7.2 Error handling conventions

- API routes that call services: catch `ModelNotFoundError`, `GalleryNotFoundError`,
  etc. and return `HTTPException(404)` or `HTTPException(400)` with a message.
- The worker: catch pipeline exceptions, call `mark_failed`, log with traceback.
- The frontend: every `fetch` is wrapped in `try/catch`; show a dismissable error
  banner in the UI.

### 7.3 Model management UI

In the left panel, below the sampler, add a collapsible "Models" section:

- Dropdown of registered models (bound to `modelKey`)
- "Scan HF cache" button → calls `/api/v2/models/scan_hf_cache`, shows candidates
  in a modal with "Register" button per candidate
- "Add local path" input → calls `/api/v2/models/scan_local_dir?path=...`
- Per registered model: rename (inline edit), delete (with confirmation)
- Optional HF API key field shown only for `source_type == "hf_repo_id"` models

---

## 13. Testing Strategy

### Principles

- **Unit tests are fast and have no I/O.** Mock the database, mock the pipeline,
  use `tmp_path` for file system tests.
- **Integration tests spin up a real in-memory database** but use a mock pipeline
  (`MagicMock` returning fake `PIL.Image` objects).
- **No test touches the real HF cache or loads a real model.**

### Running tests

```bash
# All tests
pytest tests/ -v

# Unit only (fast)
pytest tests/unit/ -v

# Integration only
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=backend --cov-report=term-missing
```

### `tests/conftest.py` — shared fixtures

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from PIL import Image as PILImage
from httpx import AsyncClient, ASGITransport

@pytest.fixture
def mock_pipeline():
    """A fake DiffusionPipeline that returns white images instantly."""
    pipeline = MagicMock()
    pipeline.return_value.images = [PILImage.new("RGB", (512, 512), "white")]
    return pipeline

@pytest.fixture
async def client(mock_pipeline, tmp_path, monkeypatch):
    """Test client with in-memory DB and mock pipeline."""
    monkeypatch.setenv("DB_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr("backend.model_registry.loader.PipelineLoader.load",
                        lambda self, entry: mock_pipeline)
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test") as c:
        yield c

def make_params(**overrides):
    """Helper: build a valid ImageGenerationParams for tests."""
    from backend.domain import ImageGenerationParams
    defaults = dict(
        prompt="test prompt", negative_prompt="", seed=42,
        width=512, height=512, steps=20, cfg=7.5,
        cfg_rescale_multiplier=0.0, sampler="euler_a", model_key="test_model"
    )
    return ImageGenerationParams(**{**defaults, **overrides})
```

### Coverage targets

| Module | Target |
|---|---|
| `queue/manager.py` | 95% |
| `pipeline/batcher.py` | 100% |
| `pipeline/preview.py` | 90% |
| `model_registry/registry.py` | 95% |
| `model_registry/scanner.py` | 85% |
| `gallery/service.py` | 90% |
| `api/*.py` | 80% (integration tests cover the rest) |

---

## 14. Running the App

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install PyTorch for your platform first (platform-specific)
# CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# MPS (Apple Silicon): pip install torch torchvision
# CPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install app dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Download Alpine.js (one-time)
mkdir -p frontend/vendor
curl -o frontend/vendor/alpine.min.js \
  https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js

# 5. Run
uvicorn backend.main:app --host 127.0.0.1 --port 8765 --reload

# 6. Open http://localhost:8765
```

---

*End of PLAN.md*
