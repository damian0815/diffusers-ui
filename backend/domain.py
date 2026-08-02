from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class GenerationCancelled(Exception):
    """Raised inside a pipeline step callback to abort an in-flight generation
    when the user requests cancellation.  Caught explicitly in
    PipelineWorker._run_batch before the generic Exception handler so that
    OOM and real errors are still properly distinguished."""

class TIStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"

@dataclass
class TIState:
    status: TIStatus = TIStatus.IDLE
    config_name: str = ""
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    steps_in_epoch: int = 0
    total_steps_in_epoch: int = 0
    train_loss: float = 0.0
    val_loss: float | None = None
    lr: float = 0.0
    elapsed_seconds: float = 0.0
    samples: list[str] = field(default_factory=list)  # validation image names
    error: str | None = None
    embedding_name: str = ""  # name of saved embedding file
    generating_samples: bool = False  # True when on-demand sample generation is running

Sampler = Literal[
    'dpmpp_sde_k', 'dpmpp_sde', 'dpmpp_2m_sde_k', 'dpmpp_2m_sde',
    'dpmpp', 'dpmpp_2m', 'dpmpp_3m', 'euler', 'euler_a', 'ddim', 'lms', 'kdpm_2_a',
    'euler_flowmatch_dynshift', 'euler_flowmatch_shift1'
]


@dataclass
class LoRAEntry:
    name: str              # display name (filename sans extension)
    filename: str          # e.g. "add_detail.safetensors"
    path: str              # absolute path on disk
    base_models: list[str] # detected compatibility ["sd-1", "sdxl", etc.]


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
    source_width: int | None = None   # pixel dimensions of the init_image
    source_height: int | None = None
    crop_left: float = 0.0          # normalized 0-1 crop rect
    crop_top: float = 0.0
    crop_width: float = 0.0         # 0 = no crop (backward compat)
    crop_height: float = 0.0
    lora_keys: list[str] = field(default_factory=list)       # filenames
    lora_strengths: list[float] = field(default_factory=list) # parallel list, same length


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
    lora_keys: tuple[str, ...] = ()


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
class MergeComponent:
    """One component of a model merge (weighted alpha)."""
    key: str          # registry key of a component model
    weight: float     # alpha (default 1.0)


@dataclass
class ModelEntry:
    key: str                             # stable UUID, generated on first registration
    name: str                            # user-facing label
    base: Literal['sd-1', 'sd-2', 'sdxl', 'sana']
    source: str                          # HF repo ID ("stabilityai/...") or absolute local path
    source_type: Literal['hf_repo_id', 'local_path']
    hf_api_key: str | None = None        # stored only if user provided one
    quantization: str | None = None      # 'int8', 'int4', or None (SANA only for now)
    merge: list[MergeComponent] | None = None   # present only on merge entries

    def __post_init__(self) -> None:
        if self.merge is not None:
            self.merge = [
                c if isinstance(c, MergeComponent) else MergeComponent(**c)
                for c in self.merge
            ]


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
        crop_left=0.0,
        crop_top=0.0,
        crop_width=0.0,
        crop_height=0.0,
    )
    return ImageGenerationParams(**{**defaults, **overrides})

