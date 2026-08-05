# Latin ASR Post-Processing (`latin-asr-postprocess`)

An end-to-end pipeline for fine-tuning BERT-style transformer models (e.g., [`latincy/latin-bert`](https://huggingface.co/latincy/latin-bert)) on **Inverse Text Normalization (ITN)**. The model accepts raw, lowercased, unpunctuated text transcriptions from Latin ASR models (such as [`njand/wav2vec2-xls-r-latin`](https://huggingface.co/njand/wav2vec2-xls-r-latin)) and restores proper casing and trailing punctuation via sequence labeling.

The training framework leverages **Modal** for cloud GPU execution, **Hugging Face Hub** for model tracking, **Weights & Biases** for experiment logging, and **CodeCarbon** for environmental impact measurement.

---

## 🛠️ Repository Layout

```text
latin-asr-postprocessing/
├── .github/
│   └── workflows/          # GitHub Actions CI configuration
├── apps/
│   ├── train.py            # Modal entrypoint: Train & evaluate on train/test split
│   └── wipe_modal_cache.py # Utility script to clean persistent Modal storage
├── latin_itn/              # Core package (dataset, models, metrics, trainer, etc.)
├── scripts/
│   ├── export_onnx.py      # ONNX export & dynamic INT8 quantization for CPU inference
│   └── infer.py            # Quick CLI tool for running inference on raw ASR text
├── tests/                  # Unit test suite
├── pyproject.toml          # PEP 621 package metadata & configuration
└── README.md

```

---

## ✨ Key Features

* **Serverless Cloud Execution:** Fine-tunes on remote NVIDIA GPUs via Modal using persistent volume storage (`latin-itn-cache`).

* **Train-Test Split Evaluation:** Trains and evaluates on the pre-existing 95% train / 5% test split (~8.46M tokens across ~292k blocks) sourced from cleaned CLTK Latin texts.

* **14-Class Joint Sequence Labeling:** Predicts combined **Casing** (`TITLE`, `LOWER`) and **Trailing Punctuation** (`NONE`, `PERIOD`, `COMMA`, `COLON`, `SEMICOLON`, `EXCLAMATION`, `QUESTION`) tags per word token.

* **Subtoken Alignment & Masking:** Accurately maps subtoken predictions back to original word boundaries using Word ID alignment, ignoring special tokens (`[CLS]`, `[SEP]`) during loss and metric calculation.

* **Decomposed Evaluation Metrics:** Logs overall Token Classification F1, Precision, and Recall, alongside decomposed metrics isolating **Casing F1** and **Punctuation F1**.

* **Telemetry & Environmental Tracking:** Logs carbon footprint via `codecarbon` and syncs offline Weights & Biases runs post-training.

---

## ⚙️ Prerequisites & Modal Secrets

To run remote training on Modal, set up the following secrets in your Modal workspace:

1. **`huggingface-secret`**: Must contain an `HF_TOKEN` environment variable with **Write permissions** to publish model artifacts, evaluation logs, and model cards.

2. **`wandb-secret`**: Must contain a `WANDB_API_KEY` environment variable for logging.

---

## ⚡ Quickstart

### 1. Local Environment Setup

Clone the repository and install the package with dev dependencies:

```bash
git clone [https://github.com/njand/latin-asr-postprocessing.git](https://github.com/njand/latin-asr-postprocessing.git)
cd latin-asr-postprocessing
pip install -e .[dev]
```

### 2. Run Tests & Linter

Run the full unit test suite and linting checks:

```bash
# Run pytest suite
pytest

# Run linter
ruff check .
```

### 3. Remote Cloud Training via Modal

Launch model training and evaluation over the dataset splits on remote Modal GPUs:

```bash
modal run apps/train.py
```

### 4. Local Inference

Run local post-processing on raw lowercased ASR output strings:

```bash
python scripts/infer.py --text "dicam plane caesar quod sentio"
# Output: "Dicam plane, Caesar, quod sentio."
```

### 5. ONNX Export & Quantization (Optional)

Export the fine-tuned model to ONNX format and apply INT8 quantization for accelerated CPU inference:

```bash
python scripts/export_onnx.py
```

### 6. Clear Remote Storage (Optional)

To wipe the persistent Modal volume cache (`/mnt/cache`):

```bash
modal run apps/wipe_modal_cache.py
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
