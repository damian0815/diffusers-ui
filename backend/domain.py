from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class GenerationCancelled(Exception):
    """Raised inside a pipeline step callback to abort an in-flight generation
    when the user requests cancellation.  Caught explicitly in
    PipelineWorker._run_batch before the generic Exception handler so that
    OOM and real errors are still properly distinguished."""

Sampler = Literal[
    'dpmpp_sde_k', 'dpmpp_sde', 'dpmpp_2m_sde_k', 'dpmpp_2m_sde',
    'dpmpp', 'dpmpp_2m', 'dpmpp_3m', 'euler', 'euler_a', 'ddim', 'lms', 'kdpm_2_a',
    'euler_flowmatch_dynshift'
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
    board_id: str | None = None  # target
    prediction_type: str = 'default'  # 'default' = don't override; else e.g. 'epsilon', 'v_prediction', 'flow_prediction'
    init_image: str | None = None   # image_name from gallery, or uploaded filename
    strength: float = 0.75          # 0.0–1.0 how much noise to add


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
    prediction_type: str = 'default'
    init_image: str | None = None


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
    # priority=True: item was enqueued to the front; skip batch window & run solo
    priority: bool = False


@dataclass
class ModelEntry:
    key: str                             # stable UUID, generated on first registration
    name: str                            # user-facing label
    base: Literal['sd-1', 'sd-2', 'sdxl', 'sana']
    source: str                          # HF repo ID ("stabilityai/...") or absolute local path
    source_type: Literal['hf_repo_id', 'local_path']
    hf_api_key: str | None = None        # stored only if user provided one
    quantization: str | None = None      # 'int8', 'int4', or None (SANA only for now)


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
        init_image=None,
        strength=0.75,
    )
    return ImageGenerationParams(**{**defaults, **overrides})

