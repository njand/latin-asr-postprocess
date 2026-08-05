from typing import Any, Tuple
from transformers import AutoModelForTokenClassification, AutoTokenizer

from latin_itn.config import DEFAULT_MODEL_NAME, ID2LABEL, LABEL2ID


def get_model_and_tokenizer(
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    revision: str | None = None,
) -> Tuple[Any, Any]:
    """Loads tokenizer and AutoModelForTokenClassification initialized for ITN schema."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=True
    )
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        revision=revision,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        attn_implementation="sdpa"
    )
    return model, tokenizer