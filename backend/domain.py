from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Sampler = Literal[
    'dpmpp_sde_k', 'dpmpp_sde', 'dpmpp_2m_sde_k', 'dpmpp_2m_sde',
    'dpmpp_2m', 'dpmpp_3m', 'euler', 'euler_a', 'ddim', 'lms', 'kdpm_2_a',
    'euler_fm_dynshift'
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
    board_id: str | None = None  # target gallery board (None = use worker default)


@dataclass(frozen=True)
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


# Simple incrementing ID (sufficient for a local app)
_counter = 0


def _next_id() -> int:
    global _counter
    _counter += 1
    return _counter


def _init_counter(min_value: int) -> None:
    """Ensure the counter never produces IDs ≤ min_value.
    Called at startup after restoring persisted items."""
    global _counter
    if min_value > _counter:
        _counter = min_value


@dataclass
class QueueItem:
    params: ImageGenerationParams
    item_id: int = field(default_factory=_next_id)
    status: ItemStatus = ItemStatus.PENDING
    output_image_name: str | None = None
    error: str | None = None


@dataclass
class ModelEntry:
    key: str                             # stable UUID, generated on first registration
    name: str                            # user-facing label
    base: Literal['sd-1', 'sd-2', 'sdxl']
    source: str                          # HF repo ID ("stabilityai/...") or absolute local path
    source_type: Literal['hf_repo_id', 'local_path']
    hf_api_key: str | None = None        # stored only if user provided one


# ---------------------------------------------------------------------------
# Test helpers (also used in app code that needs default params)
# ---------------------------------------------------------------------------

def make_params(**overrides) -> ImageGenerationParams:
    """Build a valid ImageGenerationParams with sensible defaults."""
    defaults = dict(
        prompt="test prompt",
        negative_prompt="",
        seed=42,
        width=512,
        height=512,
        steps=20,
        cfg=7.5,
        cfg_rescale_multiplier=0.0,
        sampler="euler_a",
        model_key="test_model",
    )
    return ImageGenerationParams(**{**defaults, **overrides})

