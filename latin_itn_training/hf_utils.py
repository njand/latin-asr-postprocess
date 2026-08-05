import glob
import json
import os
from typing import Any

from latin_itn_training.metrics import to_percentage


def ensure_branch_exists(api: Any, repo_id: str, branch_name: str) -> None:
    """Ensures that a target branch exists on Hugging Face Hub."""
    try:
        api.create_branch(repo_id=repo_id, branch=branch_name, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not verify/create branch '{branch_name}': {e}", flush=True)


def get_latest_checkpoint(output_dir: str) -> str | None:
    """Inspects output directory for the latest checkpoint path."""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None

    def parse_step(path: str) -> int:
        try:
            return int(path.split("checkpoint-")[-1])
        except ValueError:
            return -1

    valid_checkpoints = [ckpt for ckpt in checkpoints if parse_step(ckpt) >= 0]
    if not valid_checkpoints:
        return None

    latest_ckpt = max(valid_checkpoints, key=parse_step)
    print(f"Found local Modal checkpoint: '{latest_ckpt}'", flush=True)
    return latest_ckpt


def get_branch_progress(
    api: Any, repo_id: str, branch_name: str, hf_token: str
) -> tuple[float, float | None, float, float, float, float]:
    """Inspects target branch on HF Hub for existing progress in `trainer_state.json`."""
    from huggingface_hub import hf_hub_download

    try:
        repo_files = api.list_repo_files(repo_id=repo_id, revision=branch_name)

        if "trainer_state.json" in repo_files:
            target_filename = "trainer_state.json"
        else:
            ckpt_states = [f for f in repo_files if f.endswith("trainer_state.json")]
            if not ckpt_states:
                return 0.0, None, 0.0, 0.0, 0.0, 0.0

            def parse_ckpt_step(filename: str) -> int:
                parts = filename.split("checkpoint-")
                if len(parts) > 1:
                    try:
                        return int(parts[1].split("/")[0])
                    except ValueError:
                        return -1
                return -1

            target_filename = max(ckpt_states, key=parse_ckpt_step)

        state_file = hf_hub_download(
            repo_id=repo_id,
            filename=target_filename,
            revision=branch_name,
            token=hf_token
        )
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

        completed_epochs = float(state.get("epoch", 0.0))
        log_history = state.get("log_history", [])

        eval_logs = [log for log in log_history if "eval_loss" in log]
        if eval_logs:
            best_log = max(eval_logs, key=lambda x: x.get("eval_overall_f1", 0.0))
            best_epoch = float(best_log.get("epoch", completed_epochs))

            raw_f1 = float(best_log.get("eval_overall_f1", 0.0))
            raw_acc = float(best_log.get("eval_overall_accuracy", 0.0))
            raw_case = float(best_log.get("eval_casing_accuracy", 0.0))
            raw_punct = float(best_log.get("eval_punct_accuracy", 0.0))

            best_f1 = to_percentage(raw_f1)
            best_acc = to_percentage(raw_acc)
            best_casing_acc = to_percentage(raw_case)
            best_punct_acc = to_percentage(raw_punct)
        else:
            best_epoch = completed_epochs
            best_f1, best_acc, best_casing_acc, best_punct_acc = 0.0, 0.0, 0.0, 0.0

        return completed_epochs, best_epoch, best_f1, best_acc, best_casing_acc, best_punct_acc
    except Exception as e:
        print(f"Notice: Could not retrieve state for branch '{branch_name}': {e}", flush=True)
        return 0.0, None, 0.0, 0.0, 0.0, 0.0