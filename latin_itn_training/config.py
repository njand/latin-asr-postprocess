from dataclasses import dataclass

DEFAULT_MODEL_NAME = "latincy/latin-bert"
MAX_LENGTH = 512

# 14-Tag ITN Schema (Casing + Trailing Punctuation)
TAG_LIST = [
    "LOWER_NONE",
    "LOWER_COMMA",
    "LOWER_PERIOD",
    "LOWER_QUESTION",
    "LOWER_EXCLAMATION",
    "LOWER_SEMICOLON",
    "LOWER_COLON",
    "TITLE_NONE",
    "TITLE_COMMA",
    "TITLE_PERIOD",
    "TITLE_QUESTION",
    "TITLE_EXCLAMATION",
    "TITLE_SEMICOLON",
    "TITLE_COLON",
]

LABEL2ID = {tag: i for i, tag in enumerate(TAG_LIST)}
ID2LABEL = dict(enumerate(TAG_LIST))

PUNCT_MAP = {
    "NONE": "",
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "EXCLAMATION": "!",
    "SEMICOLON": ";",
    "COLON": ":",
}


@dataclass
class TrainingConfig:
    # --- Repos & Datasets ---
    hf_repo_id: str = "njand/latin-asr-postprocessor"
    dataset_name: str = "njand/latin-asr-post-processing-dataset"
    base_model: str = DEFAULT_MODEL_NAME
    exp_prefix: str = "v2.0"

    # --- Pipeline & Optimization ---
    epochs: int = 10
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    classifier_learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_length: int = MAX_LENGTH
    bf16: bool = True
    fp16: bool = False

    # --- Logging & Checkpointing ---
    logging_steps: int = 500
    cache_dir: str = "/mnt/cache"
    output_dir: str = "/mnt/cache/output_final"
    wandb_dir: str = "/mnt/cache/wandb"