from dataclasses import dataclass


@dataclass
class TIConfig:
    """All hyperparameters for a TI training run."""
    # Data
    gallery_id: str                    # source gallery board_id
    placeholder_token: str = "<sks>"   # new token to learn
    initializer_token: str = "dog"     # seed word for initial embedding
    num_vectors: int = 1               # multi-vector TI
    learnable_property: str = "object" # "object" or "style"
    class_prompt: str = ""             # class word appended after TI tokens
    repeats: int = 100                 # dataset repeats

    # Model
    model_key: str = ""                # key into models.json

    # Training
    resolution: int = 512
    center_crop: bool = False
    train_batch_size: int = 1
    max_train_steps: int = 500
    learning_rate: float = 5e-4
    lr_scheduler: str = "constant"     # constant, linear, cosine, etc.
    lr_warmup_steps: int = 100
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    mixed_precision: str = "bf16"        # no, fp16, bf16
    seed: int = 42

    # Validation
    validation_prompt: str = ""        # empty = skip validation
    validation_steps: int = 100
    num_validation_images: int = 4
    validation_seed: int = 42

    # Saving
    save_steps: int = 100
    output_name: str = "learned_embeds"
