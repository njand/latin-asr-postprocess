"""
Strided Sliding Window Inference Module for Latin ITN.
Processes arbitrary-length raw Latin ASR streams cleanly with CLTK pre-tokenization.
"""

from typing import Any, List, Optional
import torch

from latin_itn.cltk_tokenizer import CLTKLegacyLatinTokenizer, rejoin_cltk_enclitics_and_format
from latin_itn.models import get_model_and_tokenizer


class LatinITNPredictor:
    """High-level interface for executing inverse text normalization on Latin text."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cltk_tok: Optional[CLTKLegacyLatinTokenizer] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cltk_tok = cltk_tok or CLTKLegacyLatinTokenizer()
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls, model_path: str, device: Optional[str] = None
    ) -> "LatinITNPredictor":
        """Factory method to load model and tokenizer directly from a local path or HF Hub ID."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model, tokenizer = get_model_and_tokenizer(model_path)
        return cls(model=model, tokenizer=tokenizer, device=device)

    def predict(
        self,
        text: str,
        window_tokens: int = 250,
        stride_tokens: int = 180
    ) -> str:
        """Runs ITN inference over arbitrary long raw text using strided sliding windows."""
        return run_strided_inference(
            text=text,
            model=self.model,
            tokenizer=self.tokenizer,
            cltk_tok=self.cltk_tok,
            device=self.device,
            window_tokens=window_tokens,
            stride_tokens=stride_tokens
        )


def run_strided_inference(
    text: str,
    model: Any,
    tokenizer: Any,
    cltk_tok: CLTKLegacyLatinTokenizer,
    device: str = "cpu",
    window_tokens: int = 250,
    stride_tokens: int = 180
) -> str:
    """Runs ITN inference over arbitrary long raw text using strided sliding windows."""
    tokens_meta = cltk_tok.tokenize(text)
    if not tokens_meta:
        return ""

    total_tokens = len(tokens_meta)
    token_strings = [t[0] for t in tokens_meta]

    # Quick path for inputs fitting into a single window
    if total_tokens <= window_tokens:
        chunk_tags = _predict_chunk_tags(token_strings, model, tokenizer, device)
        return rejoin_cltk_enclitics_and_format(tokens_meta, chunk_tags)

    final_tags = ["LOWER_NONE"] * total_tokens
    margin = (window_tokens - stride_tokens) // 2

    # Track the next global index that needs a tag assignment
    next_unfilled_idx = 0

    for start_idx in range(0, total_tokens, stride_tokens):
        # If this window would overshoot the end, slide it back to catch full context
        actual_start = start_idx
        if start_idx + window_tokens > total_tokens and total_tokens > window_tokens:
            actual_start = total_tokens - window_tokens

        end_idx = min(actual_start + window_tokens, total_tokens)
        chunk_strings = token_strings[actual_start:end_idx]

        if not chunk_strings:
            break

        chunk_tags = _predict_chunk_tags(chunk_strings, model, tokenizer, device)
        is_last_chunk = end_idx == total_tokens

        # Determine local start relative to where we last left off
        local_start = max(0, next_unfilled_idx - actual_start)
        
        if is_last_chunk:
            local_end = len(chunk_strings)
        else:
            local_end = max(local_start, len(chunk_strings) - margin)

        for local_i in range(local_start, local_end):
            global_i = actual_start + local_i
            if global_i < total_tokens:
                final_tags[global_i] = chunk_tags[local_i]

        next_unfilled_idx = actual_start + local_end

        if is_last_chunk:
            break
        
    return rejoin_cltk_enclitics_and_format(tokens_meta, final_tags)


def _predict_chunk_tags(
    token_strings: List[str], model: Any, tokenizer: Any, device: str
) -> List[str]:
    """Extracts predicted tag labels for a single subtokenized chunk."""
    inputs = tokenizer(
        token_strings,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze(0).tolist()

    word_ids = inputs.word_ids(batch_index=0)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    word_predictions = []
    previous_word_idx = None

    for word_idx, pred_id in zip(word_ids, pred_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            word_predictions.append(id2label.get(pred_id, "LOWER_NONE"))
            previous_word_idx = word_idx

    while len(word_predictions) < len(token_strings):
        word_predictions.append("LOWER_NONE")

    return word_predictions