from typing import Any, Dict, Tuple
from datasets import DatasetDict, load_dataset
from latin_itn_training.config import LABEL2ID, MAX_LENGTH


def tokenize_and_align_labels(
    examples: Dict[str, Any],
    tokenizer: Any,
    max_length: int = MAX_LENGTH,
) -> Dict[str, Any]:
    """Tokenizes word sequences and aligns label tags to subtoken IDs."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    labels = []
    for i, label_tags in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                tag_str = label_tags[word_idx]
                label_ids.append(LABEL2ID.get(tag_str, -100))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def load_and_prepare_dataset(
    tokenizer: Any,
    dataset_name: str,
    hf_token: str | None = None,
    max_length: int = MAX_LENGTH,
    seed: int = 42,
    max_samples: int | None = None,
) -> Tuple[Any, Any]:
    """Loads dataset, optionally subsamples for debugging, tokenizes & aligns labels,
    drops >max_length subtoken samples, and returns (train_ds, test_ds).
    """
    raw_ds = load_dataset(dataset_name, token=hf_token)

    if max_samples:
        if isinstance(raw_ds, DatasetDict):
            for split in raw_ds:
                num_to_take = min(max_samples, len(raw_ds[split]))
                raw_ds[split] = raw_ds[split].select(range(num_to_take))
        else:
            num_to_take = min(max_samples, len(raw_ds))
            raw_ds = raw_ds.select(range(num_to_take))

    ds_train = (
        raw_ds["train"]
        if isinstance(raw_ds, DatasetDict) and "train" in raw_ds
        else (next(iter(raw_ds.values())) if isinstance(raw_ds, DatasetDict) else raw_ds)
    )

    sample_tags = set(tag for sample in ds_train["tags"] for tag in sample)
    missing_tags = sample_tags - set(LABEL2ID.keys())
    if missing_tags:
        raise ValueError(
            f"Dataset contains tags not present in config.TAG_LIST: {missing_tags}"
        )

    sample_suffix = f"s{max_samples}" if max_samples else "full"

    if isinstance(raw_ds, DatasetDict):
        tokenized_ds = DatasetDict({
            split: ds.map(
                tokenize_and_align_labels,
                batched=True,
                fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
                remove_columns=ds.column_names,
                new_fingerprint=f"latin_tok_{split}_{max_length}_{sample_suffix}",
            )
            for split, ds in raw_ds.items()
        })
    else:
        tokenized_ds = raw_ds.map(
            tokenize_and_align_labels,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
            remove_columns=raw_ds.column_names,
            new_fingerprint=f"latin_tok_{max_length}_{sample_suffix}",
        )

    initial_count = len(ds_train)
    filtered_ds = tokenized_ds.filter(
        lambda example: len(example["input_ids"]) <= max_length
    )

    train_filtered = (
        filtered_ds["train"]
        if isinstance(filtered_ds, DatasetDict) and "train" in filtered_ds
        else (next(iter(filtered_ds.values())) if isinstance(filtered_ds, DatasetDict) else filtered_ds)
    )
    dropped = initial_count - len(train_filtered)
    print(
        f"Dropped {dropped} sequences exceeding {max_length} subtokens "
        f"({dropped / max(1, initial_count):.2%})."
    )

    if isinstance(filtered_ds, DatasetDict):
        if "test" not in filtered_ds and "validation" not in filtered_ds:
            filtered_ds = filtered_ds["train"].train_test_split(
                test_size=0.05, seed=seed
            )
        elif "test" not in filtered_ds and "validation" in filtered_ds:
            filtered_ds["test"] = filtered_ds["validation"]
    else:
        filtered_ds = filtered_ds.train_test_split(test_size=0.05, seed=seed)

    return filtered_ds["train"], filtered_ds["test"]