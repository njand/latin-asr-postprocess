import gc
import os
import time
import numpy as np
from collections import Counter
from typing import Any

import torch
import torch.nn as nn

from huggingface_hub import HfApi, hf_hub_download
from transformers import Trainer, TrainerCallback, TrainingArguments, DataCollatorForTokenClassification
from transformers.trainer_callback import PrinterCallback

from latin_itn_training.config import TrainingConfig
from latin_itn_training.hf_utils import ensure_branch_exists, get_branch_progress, get_latest_checkpoint
from latin_itn_training.metrics import to_percentage
from latin_itn_training.models import get_model_and_tokenizer
from latin_itn_training.config import LABEL2ID


CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_DIM = "\033[90m"
CLR_GREEN = "\033[32m"
CLR_CYAN = "\033[36m"
CLR_YELLOW = "\033[33m"


class ModalProgressLogger(TrainerCallback):
    """Custom progress logger for Modal console output with space-track bar and ANSI colors."""

    def __init__(self, bar_width: int = 20):
        self.start_time = None
        self.bar_width = bar_width
        self.partials = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]

    @staticmethod
    def _format_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"

    def _render_bar(self, current_step: int, total_steps: int) -> str:
        total_eighths = int((current_step / max(1, total_steps)) * self.bar_width * 8)
        full_blocks = total_eighths // 8
        remainder = total_eighths % 8
        empty_blocks = self.bar_width - full_blocks - (1 if remainder else 0)

        filled = "█" * full_blocks + self.partials[remainder]
        empty = " " * max(0, empty_blocks)
        return f"{CLR_GREEN}{filled}{CLR_RESET}{empty}"

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"{CLR_BOLD}{CLR_CYAN}🚀 Training started...{CLR_RESET}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs and state.max_steps > 0:
            current_step = state.global_step
            total_steps = state.max_steps
            pct = (current_step / total_steps) * 100

            bar = self._render_bar(current_step, total_steps)
            
            now = time.time()
            elapsed = now - (self.start_time or now)
            steps_per_sec = current_step / elapsed if elapsed > 0 else 0
            eta_sec = (total_steps - current_step) / steps_per_sec if steps_per_sec > 0 else 0

            loss_val = logs.get("loss")
            loss_raw = f"{loss_val:6.4f}" if loss_val is not None else f"{'N/A':>6}"

            step_width = len(str(total_steps))

            pct_fmt = f"{CLR_BOLD}{pct:5.1f}%{CLR_RESET}"
            epoch_fmt = f"Ep {CLR_CYAN}{state.epoch or 0.0:5.2f}{CLR_RESET}"
            step_fmt = f"{CLR_CYAN}{current_step:>{step_width}d}/{total_steps}{CLR_RESET}"
            loss_fmt = f"Loss {CLR_YELLOW}{loss_raw}{CLR_RESET}"
            time_fmt = f"{CLR_DIM}{self._format_time(elapsed):>7} < ETA {self._format_time(eta_sec):>7}{CLR_RESET}"

            print(
                f"{CLR_DIM}[{CLR_RESET}{bar}{CLR_DIM}]{CLR_RESET} "
                f"{pct_fmt} | {epoch_fmt} | {step_fmt} | {loss_fmt} | {time_fmt}",
                flush=True,
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero and metrics:
            epoch = metrics.get("epoch", 0.0)
            eval_loss = metrics.get("eval_loss", 0.0)

            f1 = to_percentage(metrics.get("eval_overall_f1", 0.0))
            acc = to_percentage(metrics.get("eval_overall_accuracy", 0.0))
            prec = to_percentage(metrics.get("eval_overall_precision", 0.0))
            rec = to_percentage(metrics.get("eval_overall_recall", 0.0))

            case_acc = to_percentage(metrics.get("eval_casing_accuracy", 0.0))
            case_f1 = to_percentage(metrics.get("eval_casing_f1", 0.0))
            punct_acc = to_percentage(metrics.get("eval_punct_accuracy", 0.0))
            punct_f1 = to_percentage(metrics.get("eval_punct_f1", 0.0))

            pred_prefix = "eval_pred_dist/"
            gold_prefix = "eval_gold_dist/"
            dist_items = []

            for key, val in metrics.items():
                if key.startswith(pred_prefix):
                    label = key[len(pred_prefix) :]
                    pred_pct = to_percentage(val)
                    gold_pct = to_percentage(metrics.get(f"{gold_prefix}{label}", 0.0))
                    diff = pred_pct - gold_pct
                    dist_items.append(
                        {
                            "label": label,
                            "pred": pred_pct,
                            "gold": gold_pct,
                            "diff": diff,
                            "abs_diff": abs(diff),
                        }
                    )

            displayed_items = []
            remaining_count = 0

            if dist_items:
                by_volume = sorted(dist_items, key=lambda x: x["gold"], reverse=True)
                top_volume = by_volume[:2]
                volume_labels = {x["label"] for x in top_volume}

                remaining = [x for x in dist_items if x["label"] not in volume_labels]
                by_mismatch = sorted(remaining, key=lambda x: x["abs_diff"], reverse=True)
                top_mismatch = by_mismatch[:2]

                for item in top_volume:
                    item["tag"] = "Anchor"
                for item in top_mismatch:
                    item["tag"] = "Mismatch"

                displayed_items = top_volume + top_mismatch
                remaining_count = len(dist_items) - len(displayed_items)

            lines = [
                f"\n{CLR_BOLD}{CLR_CYAN}📊 [Eval Epoch {epoch:.1f}]{CLR_RESET} Val Loss: {CLR_YELLOW}{eval_loss:.4f}{CLR_RESET}",
                f"  ├─ {CLR_BOLD}Overall Metrics:{CLR_RESET}",
                f"  │    Macro F1: {CLR_BOLD}{f1:5.2f}%{CLR_RESET}  │  Acc: {CLR_BOLD}{acc:5.2f}%{CLR_RESET}  │  Prec: {prec:5.2f}%  │  Rec: {rec:5.2f}%",
                f"  ├─ {CLR_BOLD}Sub-Task Performance:{CLR_RESET}",
                f"  │    Casing (TITLE F1): {CLR_BOLD}{case_f1:5.2f}%{CLR_RESET}  │  Acc: {case_acc:5.2f}%",
                f"  │    Punct  (Macro F1): {CLR_BOLD}{punct_f1:5.2f}%{CLR_RESET}  │  Acc: {punct_acc:5.2f}%",
            ]

            if displayed_items:
                lines.append(
                    f"  └─ {CLR_BOLD}Class Highlights (Anchors & Top Mismatches):{CLR_RESET}"
                )
                for item in displayed_items:
                    label = item["label"]
                    p_pct = item["pred"]
                    g_pct = item["gold"]
                    diff = item["diff"]
                    tag = item["tag"]

                    diff_sign = "+" if diff > 0 else ""
                    diff_color = CLR_YELLOW if tag == "Mismatch" else ""

                    lines.append(
                        f"        • {label:<16} Pred: {CLR_BOLD}{p_pct:5.2f}%{CLR_RESET} │ Gold: {g_pct:5.2f}% │ "
                        f"Diff: {diff_color}{diff_sign}{diff:5.2f}%{CLR_RESET} [{tag}]"
                    )
                if remaining_count > 0:
                    lines.append(
                        f"        {CLR_CYAN}└─ ... and {remaining_count} other classes logged to WandB{CLR_RESET}"
                    )
            else:
                lines[3] = lines[3].replace("├─", "└─")

            print("\n".join(lines) + "\n", flush=True)


class WeightedITNTrainer(Trainer):
    """Custom Trainer penalizing majority class dominance via weighted Cross-Entropy."""
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def create_optimizer(self):
        """Initializes AdamW with differential learning rates for BERT and Classifier head."""
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.bert.named_parameters() if p.requires_grad],
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.classifier.named_parameters() if p.requires_grad],
                    "lr": getattr(self.args, "classifier_learning_rate", 1e-3),
                },
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device), 
                ignore_index=-100
            )
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_training_args(
    output_dir: str,
    config: TrainingConfig,
    num_epochs: int,
    branch_name: str,
    hf_token: str,
    run_name: str,
    is_eval_enabled: bool,
) -> TrainingArguments:
    eval_strategy = "epoch" if is_eval_enabled else "no"
    save_strategy = "epoch" if is_eval_enabled else "no"

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_accumulation_steps=10,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        num_train_epochs=num_epochs,
        logging_steps=config.logging_steps,
        disable_tqdm=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        bf16=config.bf16,
        fp16=config.fp16,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        save_total_limit=2,
        torch_compile=False,
        push_to_hub=False,
        hub_token=hf_token,
        hub_revision=branch_name,
        hub_strategy="end",
        hub_model_id=config.hf_repo_id,
        run_name=run_name,
        load_best_model_at_end=is_eval_enabled,
        metric_for_best_model="overall_f1" if is_eval_enabled else None,
        greater_is_better=True,
        report_to="wandb"
    )
    
    args.group_by_length = True
    args.classifier_learning_rate = config.classifier_learning_rate
    
    return args


def publish_model_and_readme(
    trainer: Trainer,
    api: HfApi,
    readme_content: str,
    output_dir: str,
    hf_repo_id: str,
    branch_name: str,
    commit_msg: str,
    base_model_name: str = "latincy/latin-bert",
) -> None:
    trainer.save_model(output_dir)

    tokenizer_assets = [
        "tokenizer_config.json",
        "latin.subword.encoder",
        "tokenization_latin_bert.py",
        "tokenization_latin_bert_fast.py",
        "special_tokens_map.json",
    ]

    for filename in tokenizer_assets:
        try:
            hf_hub_download(
                repo_id=base_model_name,
                filename=filename,
                local_dir=output_dir,
            )
        except Exception:
            pass

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    api.upload_folder(
        folder_path=output_dir,
        repo_id=hf_repo_id,
        revision=branch_name,
        commit_message=commit_msg,
        ignore_patterns=["checkpoint-*"],
    )


def parse_emissions_summary(emissions_csv_path: str = "emissions.csv") -> tuple[str, str]:
    if not os.path.exists(emissions_csv_path):
        return "N/A", "N/A"

    try:
        import pandas as pd
        df = pd.read_csv(emissions_csv_path)
        total_duration_sec = df["duration"].sum() if "duration" in df else 0.0
        total_emissions_kg = df["emissions"].sum() if "emissions" in df else 0.0

        return f"{total_duration_sec / 3600.0:.2f} hours", f"{total_emissions_kg:.4f} kg CO2eq"
    except Exception:
        return "N/A", "N/A"


def execute_training_step(
    config: TrainingConfig,
    branch_name: str,
    run_name: str,
    target_epochs: int,
    output_dir: str,
    train_ds: Any,
    eval_ds: Any | None,
    tokenizer: Any,
    compute_metrics: Any | None,
    callbacks: list[Any] | None,
    api: HfApi,
    hf_token: str,
    readme_content_fn: Any,
    commit_message: str
) -> dict[str, Any]:
    import wandb

    if wandb.run is not None:
        wandb.finish()

    wandb.init(
        project="latin-itn",
        id=run_name.replace(".", "_"),
        name=run_name,
        reinit="finish_previous"
    )

    ensure_branch_exists(api, config.hf_repo_id, branch_name)
    completed_epochs, best_epoch, saved_f1, saved_acc, saved_casing_acc, saved_punct_acc = get_branch_progress(
        api, config.hf_repo_id, branch_name, hf_token
    )

    if completed_epochs >= target_epochs:
        print(f"-> Branch '{branch_name}' previously completed. Preserving metrics.", flush=True)
        return {
            "completed_previously": True,
            "best_epoch": best_epoch if best_epoch is not None else target_epochs,
            "f1": saved_f1,
            "acc": saved_acc,
            "casing_acc": saved_casing_acc,
            "punct_acc": saved_punct_acc
        }

    resume_ckpt = get_latest_checkpoint(output_dir)
    model_checkpoint = config.hf_repo_id if (completed_epochs > 0 and not resume_ckpt) else config.base_model
    revision_target = branch_name if (completed_epochs > 0 and not resume_ckpt) else None

    model, _ = get_model_and_tokenizer(model_checkpoint, revision=revision_target)

    is_eval_enabled = eval_ds is not None
    training_args = build_training_args(
        output_dir=output_dir,
        config=config,
        num_epochs=target_epochs,
        branch_name=branch_name,
        hf_token=hf_token,
        run_name=run_name,
        is_eval_enabled=is_eval_enabled,
    )

    tokenizer.pre_tokenizer = None
    tokenizer.save_pretrained = lambda *args, **kwargs: None

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8 if config.bf16 or config.fp16 else None
    )

    all_labels = [label for example in train_ds["labels"] for label in example if label != -100]
    counts = Counter(all_labels)
    total_samples = sum(counts.values())
    num_classes = len(LABEL2ID)

    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        w = np.sqrt(total_samples / (num_classes * count))
        weights.append(w)

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()

    weights = np.clip(weights, a_min=0.2, a_max=3.0)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    trainer = WeightedITNTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if is_eval_enabled else None,
        callbacks=callbacks
    )

    trainer.remove_callback(PrinterCallback)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    eval_results = {}
    if is_eval_enabled:
        eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
        if eval_logs:
            best_log = max(
                eval_logs,
                key=lambda x: x.get("eval_overall_f1", 0.0)
            )
            eval_results["f1"] = to_percentage(best_log.get("eval_overall_f1", 0.0))
            eval_results["acc"] = to_percentage(best_log.get("eval_overall_accuracy", 0.0))
            eval_results["casing_acc"] = to_percentage(best_log.get("eval_casing_accuracy", 0.0))
            eval_results["punct_acc"] = to_percentage(best_log.get("eval_punct_accuracy", 0.0))

    readme_text = readme_content_fn(trainer, eval_results)
    publish_model_and_readme(
        trainer=trainer,
        api=api,
        readme_content=readme_text,
        output_dir=training_args.output_dir,
        hf_repo_id=config.hf_repo_id,
        branch_name=branch_name,
        commit_msg=commit_message
    )

    _, updated_best_epoch, _, _, _, _ = get_branch_progress(api, config.hf_repo_id, branch_name, hf_token)

    if wandb.run is not None:
        wandb.finish()

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "completed_previously": False,
        "best_epoch": updated_best_epoch if updated_best_epoch is not None else target_epochs,
        "f1": eval_results.get("f1", 0.0),
        "acc": eval_results.get("acc", 0.0),
        "casing_acc": eval_results.get("casing_acc", 0.0),
        "punct_acc": eval_results.get("punct_acc", 0.0)
    }