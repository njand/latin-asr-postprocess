from collections import Counter
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from latin_itn_training.config import ID2LABEL


def to_percentage(raw_val: float) -> float:
    """Converts a raw ratio metric to percentage (0.85 -> 85.0)."""
    return float(raw_val) * 100.0


def _parse_tag(tag: str) -> Tuple[str, str]:
    """Extracts casing and punctuation components from a tag string."""
    parts = tag.split("_", 1)
    casing = parts[0] if parts else "LOWER"
    punct = parts[1] if len(parts) > 1 else "NONE"
    return casing, punct


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Computes overall accuracy, macro precision/recall/F1, split casing/punct metrics,
    and predicted vs. gold class distributions.
    """
    predictions, labels = eval_pred

    # Unpack predictions if model returns extra outputs (e.g., hidden states)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=2)

    flat_preds = []
    flat_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                flat_preds.append(ID2LABEL[pred])
                flat_labels.append(ID2LABEL[label])

    metrics: Dict[str, float] = {}

    # Handle edge case where all labels are masked
    if not flat_labels:
        metrics.update({
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "overall_f1": 0.0,
            "overall_accuracy": 0.0,
            "casing_accuracy": 0.0,
            "casing_f1": 0.0,
            "punct_accuracy": 0.0,
            "punct_f1": 0.0,
        })
        for label_name in ID2LABEL.values():
            metrics[f"pred_dist/{label_name}"] = 0.0
            metrics[f"gold_dist/{label_name}"] = 0.0
        return metrics

    # --- Overall Metrics ---
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="macro", zero_division=0
    )
    accuracy = float(np.mean(np.array(flat_preds) == np.array(flat_labels)))

    # --- Split Sub-task Metrics (Casing & Punctuation) ---
    casing_preds = [_parse_tag(pred)[0] for pred in flat_preds]
    casing_labels = [_parse_tag(label)[0] for label in flat_labels]
    casing_acc = float(np.mean(np.array(casing_preds) == np.array(casing_labels)))
    casing_f1 = float(
        f1_score(casing_labels, casing_preds, pos_label="TITLE", average="binary", zero_division=0)
    )

    punct_preds = [_parse_tag(pred)[1] for pred in flat_preds]
    punct_labels = [_parse_tag(label)[1] for label in flat_labels]
    punct_acc = float(np.mean(np.array(punct_preds) == np.array(punct_labels)))
    _, _, punct_f1, _ = precision_recall_fscore_support(
        punct_labels, punct_preds, average="macro", zero_division=0
    )

    metrics.update({
        "overall_precision": float(precision),
        "overall_recall": float(recall),
        "overall_f1": float(f1),
        "overall_accuracy": accuracy,
        "casing_accuracy": casing_acc,
        "casing_f1": float(casing_f1),
        "punct_accuracy": punct_acc,
        "punct_f1": float(punct_f1),
    })

    # --- Class Distributions (Predicted vs. Gold Ground Truth) ---
    total_tokens = len(flat_preds)
    pred_counts = Counter(flat_preds)
    gold_counts = Counter(flat_labels)

    # Iterating over fixed ID2LABEL keys ensures consistent WandB metric keys every run
    for label_name in ID2LABEL.values():
        metrics[f"pred_dist/{label_name}"] = float(pred_counts[label_name] / total_tokens)
        metrics[f"gold_dist/{label_name}"] = float(gold_counts[label_name] / total_tokens)

    return metrics