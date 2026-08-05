# ruff: noqa: E402
# /// script
# dependencies = [
#     "optimum[onnxruntime]>=1.16.0",
#     "onnx>=1.14.0",
#     "transformers>=4.38.0",
#     "torch>=2.0.0",
#     "datasets>=2.16.0",
#     "numpy>=1.24.0",
#     "tqdm>=4.66.0",
#     "scikit-learn>=1.2.0",
# ]
# ///

"""
Export, dynamic INT8 quantize, benchmark, and publish fine-tuned 
Latin ITN token classification models to ONNX format for accelerated CPU inference.
"""

import logging
import os
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root (parent of scripts/) is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from latin_itn.config import ID2LABEL
from latin_itn.metrics import compute_metrics, to_percentage

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_onnx")


@dataclass
class PipelineConfig:
    """Holds pipeline execution configuration options."""

    model_id: str = "njand/latin-asr-postprocessor"
    dataset_id: str = "njand/latin-asr-post-processing-dataset"
    output_dir: Path = field(default_factory=lambda: Path("./onnx_export"))
    arch: str = "avx2"
    max_samples: int | None = None
    push_to_hub: bool = False
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))


@dataclass
class BenchmarkMetric:
    """Encapsulates execution and evaluation metrics for a single model variant."""

    variant: str
    size_mb: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_seq_sec: float
    overall_acc: float
    overall_f1: float
    casing_f1: float
    punct_f1: float

    def to_markdown_row(self) -> str:
        return (
            f"| {self.variant} "
            f"| {self.size_mb:.1f} MB "
            f"| {self.mean_latency_ms:.2f} ms "
            f"| {self.p50_latency_ms:.2f} ms "
            f"| {self.p95_latency_ms:.2f} ms "
            f"| {self.p99_latency_ms:.2f} ms "
            f"| {self.throughput_seq_sec:.1f} seq/s "
            f"| {self.overall_acc:.2f}% "
            f"| {self.overall_f1:.2f}% "
            f"| {self.casing_f1:.2f}% "
            f"| {self.punct_f1:.2f}% |"
        )


class ONNXExporter:
    """Handles PyTorch-to-ONNX conversion and quantization."""

    ARCH_CONFIG_MAP = {
        "avx2": AutoQuantizationConfig.avx2,
        "avx512": AutoQuantizationConfig.avx512,
        "arm64": AutoQuantizationConfig.arm64,
    }

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def export_fp32(self) -> Path:
        """Exports PyTorch model and tokenizer configs to ONNX FP32."""
        logger.info("Exporting PyTorch model '%s' to ONNX FP32...", self.config.model_id)

        model = ORTModelForTokenClassification.from_pretrained(
            self.config.model_id,
            export=True,
            trust_remote_code=True,
        )
        model.save_pretrained(self.config.output_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            use_fast=True
        )
        try:
            tokenizer.save_pretrained(self.config.output_dir)
        except Exception as e:
            logger.warning("Skipped saving tokenizer artifacts to %s: %s", self.config.output_dir, e)
            
        fp32_path = self.config.output_dir / "model.onnx"
        size_mb = fp32_path.stat().st_size / 1e6
        logger.info("Saved ONNX FP32 model: %s (%.1f MB)", fp32_path, size_mb)
        return fp32_path

    def quantize_int8(self) -> Path:
        """Applies dynamic INT8 quantization to exported ONNX model."""
        logger.info("Quantizing ONNX model to INT8 (profile: %s)...", self.config.arch)

        config_factory = self.ARCH_CONFIG_MAP.get(
            self.config.arch.lower(), AutoQuantizationConfig.avx2
        )
        qconfig = config_factory(
            is_static=False,
            per_channel=False,
            operators_to_quantize=["MatMul"],
        )

        quantizer = ORTQuantizer.from_pretrained(
            self.config.output_dir,
            file_name="model.onnx",
        )
        quantizer.quantize(
            save_dir=self.config.output_dir,
            quantization_config=qconfig,
            file_suffix="quantized",
        )

        int8_path = self.config.output_dir / "model_quantized.onnx"
        size_mb = int8_path.stat().st_size / 1e6
        logger.info("Saved INT8 model: %s (%.1f MB)", int8_path, size_mb)
        return int8_path


class BenchmarkEvaluator:
    """Handles dataset loading, inference benchmark, and metrics calculation."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _prepare_inputs(
        self, sample: dict[str, Any], tokenizer: PreTrainedTokenizerBase
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        """Prepares pre-tokenized word inputs and extracts ground-truth label IDs."""
        tokens = sample.get("tokens") or sample.get("words")
        if tokens is None and "text" in sample:
            tokens = sample["text"].split()
        elif tokens is None:
            tokens = []

        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )

        raw_labels = sample.get("labels") or sample.get("ner_tags") or sample.get("tags") or []
        label2id = getattr(tokenizer, "label2id", None) or getattr(self, "_label2id", {})

        label_ids = []
        for label in raw_labels:
            if isinstance(label, str):
                label_ids.append(label2id.get(label, -100))
            elif isinstance(label, int):
                label_ids.append(label)

        return inputs, label_ids

    def _evaluate_model_variant(
        self,
        label: str,
        filename: str,
        dataset,
        tokenizer: PreTrainedTokenizerBase,
    ) -> BenchmarkMetric | None:
        model_path = self.config.output_dir / filename
        if not model_path.exists():
            logger.warning("Skipping %s: file %s not found.", label, filename)
            return None

        size_mb = model_path.stat().st_size / 1e6
        logger.info("Benchmarking %s (%s) over %d samples...", label, filename, len(dataset))

        model = ORTModelForTokenClassification.from_pretrained(
            self.config.output_dir, file_name=filename
        )

        self._label2id = getattr(model.config, "label2id", {})

        # Warmup pass matching strided input shape
        warmup_tokens = ["dicam", "plane", "caesar", "quod", "sentio"]
        warmup_inputs = tokenizer(warmup_tokens, is_split_into_words=True, return_tensors="pt")
        _ = model(**warmup_inputs)

        latencies = []
        all_aligned_logits = []
        all_aligned_labels = []

        for sample in tqdm(
            dataset,
            desc=f"Evaluating {label}",
            unit="sample",
            leave=True,
        ):
            inputs, target_label_ids = self._prepare_inputs(sample, tokenizer)
            if not inputs.get("input_ids").numel():
                continue

            start = time.perf_counter()
            outputs = model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000.0)

            # Subword-to-word alignment for compute_metrics
            word_ids = inputs.word_ids(batch_index=0)
            logits_tensor = outputs.logits.squeeze(0).detach().cpu().numpy()

            seq_logits = []
            seq_labels = []
            previous_word_idx = None

            for subword_idx, word_idx in enumerate(word_ids):
                if word_idx is not None and word_idx != previous_word_idx:
                    seq_logits.append(logits_tensor[subword_idx])
                    if target_label_ids and word_idx < len(target_label_ids):
                        seq_labels.append(target_label_ids[word_idx])
                    else:
                        seq_labels.append(-100)
                    previous_word_idx = word_idx

            if seq_logits:
                all_aligned_logits.append(seq_logits)
                all_aligned_labels.append(seq_labels)

        # Pad sequences to array shape (N, max_len, num_classes) for compute_metrics
        eval_metrics = {}
        if all_aligned_labels:
            max_len = max(len(seq) for seq in all_aligned_labels)
            num_classes = len(ID2LABEL)
            num_samples = len(all_aligned_labels)

            padded_logits = np.zeros((num_samples, max_len, num_classes), dtype=np.float32)
            padded_labels = np.full((num_samples, max_len), -100, dtype=np.int64)

            for i, (seq_lgt, seq_lbl) in enumerate(zip(all_aligned_logits, all_aligned_labels)):
                seq_len = len(seq_lbl)
                padded_logits[i, :seq_len, :] = np.array(seq_lgt)
                padded_labels[i, :seq_len] = np.array(seq_lbl)

            eval_metrics = compute_metrics((padded_logits, padded_labels))

        # Latency statistics
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        p99_lat = float(np.percentile(latencies, 99)) if latencies else 0.0
        total_time_sec = sum(latencies) / 1000.0
        throughput = (len(dataset) / total_time_sec) if total_time_sec > 0 else 0.0

        # Classification metrics
        overall_acc = to_percentage(eval_metrics.get("overall_accuracy", 0.0))
        overall_f1 = to_percentage(eval_metrics.get("overall_f1", 0.0))
        casing_f1 = to_percentage(eval_metrics.get("casing_f1", 0.0))
        punct_f1 = to_percentage(eval_metrics.get("punct_f1", 0.0))

        return BenchmarkMetric(
            variant=label,
            size_mb=size_mb,
            mean_latency_ms=mean_lat,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            throughput_seq_sec=throughput,
            overall_acc=overall_acc,
            overall_f1=overall_f1,
            casing_f1=casing_f1,
            punct_f1=punct_f1,
        )

    def run(self) -> None:
        """Executes benchmarks across exported FP32 and INT8 ONNX models."""
        logger.info("Loading test dataset '%s'...", self.config.dataset_id)
        dataset = load_dataset(self.config.dataset_id, split="test")

        if self.config.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.config.max_samples)))

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.output_dir,
            trust_remote_code=True,
            use_fast=True
        )

        models_to_test = [
            ("FP32 Model", "model.onnx"),
            ("INT8 Quantized", "model_quantized.onnx"),
        ]

        results = []
        for label, filename in models_to_test:
            metric = self._evaluate_model_variant(label, filename, dataset, tokenizer)
            if metric:
                results.append(metric)

        self._render_markdown_table(results)

    @staticmethod
    def _render_markdown_table(results: list[BenchmarkMetric]) -> None:
        print("\n" + "=" * 105)
        print("### ONNX Model Detailed Benchmark Results")
        print("=" * 105)
        print("| Variant | Size | Mean Lat | P50 Lat | P95 Lat | P99 Lat | Throughput | Overall Acc | Overall F1 | Casing F1 | Punct F1 |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in results:
            print(r.to_markdown_row())
        print("=" * 105 + "\n")


class HubPublisher:
    """Handles publishing artifacts to Hugging Face Hub."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def publish(self) -> None:
        logger.info("Uploading ONNX models to Hugging Face Hub repo '%s'...", self.config.model_id)
        api = HfApi(token=self.config.hf_token)
        api.upload_folder(
            folder_path=str(self.config.output_dir),
            repo_id=self.config.model_id,
            repo_type="model",
            allow_patterns=["*.onnx", "*.json", "tokenizer*"],
        )
        logger.info("Uploaded successfully to https://huggingface.co/%s", self.config.model_id)


def parse_args() -> PipelineConfig:
    parser = ArgumentParser(description="Export and quantize Latin ITN models to ONNX.")
    parser.add_argument(
        "--model-id",
        default="njand/latin-asr-postprocessor",
        help="Hugging Face repo ID or local checkpoint path.",
    )
    parser.add_argument(
        "--dataset-id",
        default="njand/latin-asr-post-processing-dataset",
        help="Hugging Face dataset repo ID for benchmarking.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./onnx_export"),
        help="Output directory for exported ONNX files.",
    )
    parser.add_argument(
        "--arch",
        choices=["avx2", "avx512", "arm64"],
        default="avx2",
        help="Target CPU instruction set profile for quantization.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of test samples to benchmark.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Publish exported ONNX models to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="HF write token (defaults to $HF_TOKEN env var or cached HF login token).",
    )

    args = parser.parse_args()
    return PipelineConfig(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        arch=args.arch,
        max_samples=args.max_samples,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
    )


def main():
    config = parse_args()

    # 1. Export & Quantize
    exporter = ONNXExporter(config)
    exporter.export_fp32()
    exporter.quantize_int8()

    # 2. Benchmark
    evaluator = BenchmarkEvaluator(config)
    evaluator.run()

    # 3. Publish (Optional)
    if config.push_to_hub:
        publisher = HubPublisher(config)
        publisher.publish()


if __name__ == "__main__":
    main()