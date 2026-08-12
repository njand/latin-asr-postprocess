from typing import Any

from latin_itn_training.metrics import to_percentage


def build_readme_table(log_history: list[dict[str, Any]]) -> str:
    """Constructs a formatted Markdown metrics table from trainer log history."""
    train_losses = {entry["step"]: entry["loss"] for entry in log_history if "loss" in entry}
    eval_metrics = {entry["step"]: entry for entry in log_history if "eval_loss" in entry}

    if not eval_metrics:
        return ""

    best_step = max(eval_metrics.keys(), key=lambda s: eval_metrics[s].get("eval_overall_f1", 0.0))

    table_rows = []
    for step in sorted(eval_metrics.keys()):
        e = eval_metrics[step]
        epoch_val = round(e.get("epoch", 0.0), 1)
        epoch_str = f"{epoch_val:g}"
        val_loss = e.get("eval_loss", 0.0)

        f1_raw = e.get("eval_overall_f1", 0.0)
        acc = to_percentage(e.get("eval_overall_accuracy", 0.0))
        casing_acc = to_percentage(e.get("eval_casing_accuracy", 0.0))
        punct_acc = to_percentage(e.get("eval_punct_accuracy", 0.0))

        prior_steps = [s for s in train_losses if s <= step]
        tr_str = f"{train_losses[max(prior_steps)]:.4f}" if prior_steps else "N/A"

        cols = [
            epoch_str,
            tr_str,
            f"{val_loss:.4f}",
            f"{f1_raw:.4f}",
            f"{acc:.2f}%",
            f"{casing_acc:.2f}%",
            f"{punct_acc:.2f}%",
        ]

        if step == best_step:
            cols = [f"**{c}**" for c in cols]

        table_rows.append(f"| {' | '.join(cols)} |")

    header = (
        "| Epoch | Train Loss | Val Loss | Overall F1 | Overall Acc | Casing Acc | Punct Acc |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    return header + "\n" + "\n".join(table_rows)


def generate_final_readme(
    base_model: str,
    dataset_name: str,
    hf_repo_id: str,
    eval_results: dict[str, Any] | None,
    hours_str: str,
    carbon_str: str,
    table_md: str,
) -> str:
    """Generates the Model Card README string for the standard 95/5 training run."""
    eval_results = eval_results or {}

    # Raw decimal metrics
    raw_f1 = float(eval_results.get("eval_overall_f1", 0.0))
    raw_acc = float(eval_results.get("eval_overall_accuracy", 0.0))
    raw_precision = float(eval_results.get("eval_overall_precision", eval_results.get("eval_precision", 0.0)))
    raw_recall = float(eval_results.get("eval_overall_recall", eval_results.get("eval_recall", 0.0)))
    val_loss = float(eval_results.get("eval_loss", 0.0))

    raw_casing_acc = float(eval_results.get("eval_casing_accuracy", 0.0))
    raw_casing_f1 = float(eval_results.get("eval_casing_f1", 0.0))
    raw_punct_acc = float(eval_results.get("eval_punct_accuracy", 0.0))
    raw_punct_f1 = float(eval_results.get("eval_punct_f1", 0.0))

    # Display percentage metrics
    acc_pct = to_percentage(raw_acc)
    precision_pct = to_percentage(raw_precision)
    recall_pct = to_percentage(raw_recall)
    casing_acc_pct = to_percentage(raw_casing_acc)
    punct_acc_pct = to_percentage(raw_punct_acc)

    epoch_val = eval_results.get("epoch")
    epoch_header = f" (Epoch {int(round(epoch_val))} - Best Checkpoint)" if epoch_val is not None else ""
    epoch_text = f"Epoch {int(round(epoch_val))}" if epoch_val is not None else "the best checkpoint"

    return f"""---
language:
- la
license: mit
tags:
- latin
- token-classification
- inverse-text-normalization
- casing
- punctuation
- capitalization
- punctuation-restoration
- asr-post-processing
- classical-latin
pipeline_tag: token-classification
datasets:
- {dataset_name}
model-index:
- name: Latin ASR Post-Processor
  results:
  - task:
      type: token-classification
      name: Inverse Text Normalization
    dataset:
      type: {dataset_name}
      name: Latin ASR Post-Processing Dataset
    metrics:
    - name: Macro F1
      type: f1
      value: {raw_f1:.4f}
    - name: Overall Accuracy
      type: accuracy
      value: {raw_acc:.4f}
    - name: Precision
      type: precision
      value: {raw_precision:.4f}
    - name: Recall
      type: recall
      value: {raw_recall:.4f}
base_model:
- {base_model}
---

# Latin ASR Post-Processor (Casing & Punctuation Restoration)

An Inverse Text Normalization (ITN) transformer model fine-tuned to convert unformatted, raw Latin Automatic Speech Recognition (ASR) outputs into fully formatted, classical Latin text. It simultaneously restores capitalization and trailing punctuation using a **14-class composite sequence labeling schema**.

---

### 📌 Quick Links
- **Live Demo:** [Gradio Interface](https://huggingface.co/spaces/njand/latin-asr-demo)
- **Source Code:** [GitHub Repository](https://github.com/njand/latin-asr-postprocess)
- **Base Model:** [`{base_model}`](https://huggingface.co/{base_model})
- **Dataset:** [`{dataset_name}`](https://huggingface.co/datasets/{dataset_name})

---

## 🛠️ Pipeline Architecture & Preprocessing

This model is intended to be used directly downstream of the acoustic model [`njand/wav2vec2-xls-r-latin`](https://huggingface.co/njand/wav2vec2-xls-r-latin). 

Because raw ASR models emit stream-of-consciousness text (lowercased, space-separated, and unpunctuated), the text must pass through an input normalization pipeline before being fed into this model for casing and punctuation restoration.

```text
+-----------------------+     +-------------------------------+     +-------------------------------+
|  Raw Audio Waveform   | --> |  njand/wav2vec2-xls-r-latin   | --> | Preprocessing & Normalization |
+-----------------------+     +-------------------------------+     +-------------------------------+
                                                                                    |
                                                                                    v
+-----------------------+     +-------------------------------+     +-------------------------------+
| Formatted Text Output | <-- |   Latin ASR Post-Processor    | <-- |   Custom CLTK Tokenization     |
+-----------------------+     +-------------------------------+     +-------------------------------+

```

### Input Preprocessing Requirements

To prepare raw transcript outputs for inference, apply the following sequence of transformations:

1. **Macron Stripping:** Remove all vowel length diacritics (e.g., *ā, ē, ī, ō, ū, ȳ* → *a, e, i, o, u, y*).
2. **Orthographic Standardization:** Standardize consonant/vowel variants:
* Convert *j* → *i* and *v* → *u*.
* Handle orthographic exceptions (e.g., *ejicio* → *eicio*).


3. **Custom CLTK Word Tokenization:** Run the normalized string through a version of the **[CLTK (Classical Language Toolkit v0)](https://github.com/cltk/cltk/tree/v0/cltk/tokenize/latin)** Latin word tokenizer.
> *Note:* Because official CLTK v0 tokenization scripts are unmaintained, a bespoke implementation of the tokenizer was executed dynamically during training preprocessing rather than being pre-applied to the static dataset.



---

## 🚀 Quickstart & Inference Utility

Below is a complete Python script demonstrating how to prepare raw ASR output and run inference using the post-processing pipeline.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

PUNCT_MAP = {{
    "NONE": "",
    "COMMA": ",",
    "PERIOD": ".",
    "SEMICOLON": ";",
    "COLON": ":",
    "QUESTION": "?",
    "EXCLAMATION": "!",
}}

def format_token(word: str, tag: str) -> str:
    \"\"\"Applies composite ITN tag (e.g., 'TITLE_COMMA') to a word token.\"\"\"
    parts = tag.split("_")
    if len(parts) != 2:
        return word

    casing, punct = parts[0], parts[1]

    if casing == "TITLE":
        word = word.capitalize()
    elif casing == "LOWER":
        word = word.lower()

    return f"{{word}}{{PUNCT_MAP.get(punct, '')}}"

def restore_latin_text(pipe, raw_text: str) -> str:
    \"\"\"Runs inference and reconstructs formatted Latin text.\"\"\"
    predictions = pipe(raw_text, aggregation_strategy="first")
    formatted_words = [
        format_token(pred["word"].strip(" "), pred["entity_group"])
        for pred in predictions
    ]
    return " ".join(formatted_words)

# 1. Load pipeline
model_id = "{hf_repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

itn_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)

# 2. Test reconstruction with preprocessed ASR output
raw_asr_input = "gallia est omnis divisa in partes tres quarum unam incolunt belgae"
print(restore_latin_text(itn_pipe, raw_asr_input))
# Output: "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae."

```

---

## 🏷️ Composite Label Schema

Target labels utilize a **14-class composite sequence schema** that pairs Casing state with Trailing Punctuation state:

$$\text{{Label}} = \text{{Casing}} \times \text{{Punctuation}}$$

* **Casing Tags (2):** `LOWER`, `TITLE`
* **Punctuation Tags (7):** `NONE`, `COMMA`, `PERIOD`, `SEMICOLON`, `COLON`, `QUESTION`, `EXCLAMATION`

---

## 📊 Benchmarks & Performance{epoch_header}

Evaluated on a 95/5 train/holdout split across diverse Classical Latin literary and historical corpora.

### Overall Summary Metrics

| Metric | Score |
| --- | --- |
| **Overall Accuracy** | **{acc_pct:.2f}%** |
| **Macro F1** | **{raw_f1:.4f}** |
| **Precision** | **{precision_pct:.2f}%** |
| **Recall** | **{recall_pct:.2f}%** |
| **Validation Loss** | **{val_loss:.4f}** |

### 🎯 Sub-Task Breakdown

| Task | Accuracy | F1 Score |
| --- | --- | --- |
| **Casing Restoration** | **{casing_acc_pct:.2f}%** | **{raw_casing_f1:.4f}** |
| **Punctuation Insertion** | **{punct_acc_pct:.2f}%** | **{raw_punct_f1:.4f}** |

---

## 📈 Training Progression

The model was fine-tuned from [`{base_model}`](https://www.google.com/search?q=https://huggingface.co/%7Bbase_model%7D). Model weights from {epoch_text} were selected based on optimal overall F1.

{table_md if table_md else "*No evaluation steps recorded during training.*"}

---

## ⚡ Hardware & Environmental Footprint

* **Hardware Infrastructure:** NVIDIA L4 GPU via Modal
* **Training Time:** {hours_str}
* **Estimated Carbon Emissions:** {carbon_str}
"""
