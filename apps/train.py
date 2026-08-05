import os
import subprocess
from typing import Any

import modal

from latin_itn.models import get_model_and_tokenizer
from latin_itn_training.config import TrainingConfig
from latin_itn_training.dataset import load_and_prepare_dataset
from latin_itn_training.metrics import compute_metrics
from latin_itn_training.templates import build_readme_table, generate_final_readme
from latin_itn_training.trainer_utils import (
    ModalProgressLogger,
    execute_training_step,
    parse_emissions_summary,
)

app = modal.App("latin-itn-pipeline")

training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "scikit-learn",
        "huggingface_hub",
        "pandas",
        "numpy",
        "wandb",
        "codecarbon",
        "sentencepiece",
        "protobuf",
    )
    .add_local_python_source("latin_itn")
)

cache_volume = modal.Volume.from_name("latin-itn-cache", create_if_missing=True)

CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_BG_MAGENTA = "\033[30;45m"


def run_training_job(
    config: TrainingConfig,
    train_ds: Any,
    eval_ds: Any,
    tokenizer: Any,
    api: Any,
    hf_token: str,
    tracker: Any,
) -> dict:
    from transformers import EarlyStoppingCallback

    print(
        f"\n{CLR_BOLD}{CLR_BG_MAGENTA} =================== STARTING TRAINING (95/5 SPLIT) =================== {CLR_RESET}",
        flush=True,
    )

    def readme_builder(trainer, eval_res):
        if tracker is not None:
            tracker.flush()

        hours_str, carbon_str = parse_emissions_summary()
        table_md = build_readme_table(trainer.state.log_history) if trainer else ""

        return generate_final_readme(
            base_model=config.base_model,
            dataset_name=config.dataset_name,
            hf_repo_id=config.hf_repo_id,
            eval_results=eval_res,
            hours_str=hours_str,
            carbon_str=carbon_str,
            table_md=table_md,
        )

    res = execute_training_step(
        config=config,
        branch_name="main",
        run_name=f"{config.exp_prefix}-run",
        target_epochs=config.epochs,
        output_dir=config.output_dir,
        train_ds=train_ds,
        eval_ds=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            ModalProgressLogger()
        ],
        api=api,
        hf_token=hf_token,
        readme_content_fn=readme_builder,
        commit_message="Completed ITN training on 95/5 split",
    )

    cache_volume.commit()
    return res


@app.function(
    image=training_image,
    gpu="L4",
    timeout=86400,
    volumes={"/mnt/cache": cache_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def run_pipeline(config: TrainingConfig | None = None):
    from codecarbon import EmissionsTracker
    from huggingface_hub import HfApi

    if config is None:
        config = TrainingConfig()

    if "WANDB_API_KEY" not in os.environ:
        raise ValueError("WANDB_API_KEY secret was not found in environment!")

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = config.wandb_dir
    os.makedirs(config.wandb_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not found. Ensure Modal secret is attached."
        )

    api = HfApi(token=hf_token)
    tracker = EmissionsTracker(
        output_dir=".", output_file="emissions.csv", log_level="warning"
    )
    tracker.start()

    try:
        _, tokenizer = get_model_and_tokenizer(config.base_model)
        
        train_ds, eval_ds = load_and_prepare_dataset(
            tokenizer,
            config.dataset_name,
            hf_token,
            max_length=config.max_length
        )

        run_training_job(
            config=config,
            train_ds=train_ds,
            eval_ds=eval_ds,
            tokenizer=tokenizer,
            api=api,
            hf_token=hf_token,
            tracker=tracker
        )

    finally:
        try:
            tracker.stop()
        except Exception as e:
            print(f"Failed to stop carbon tracker: {e}")

    print("\n☁️ Syncing offline W&B runs to cloud...", flush=True)
    try:
        subprocess.run(
            ["wandb", "sync", "--sync-all"],
            cwd=config.wandb_dir,
            check=True,
        )
        print("✅ W&B sync complete.", flush=True)
    except Exception as e:
        print(f"⚠️ W&B sync failed: {e}", flush=True)


@app.local_entrypoint()
def main():
    run_pipeline.spawn()