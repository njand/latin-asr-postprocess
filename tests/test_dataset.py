from unittest.mock import MagicMock

import pytest

from latin_itn_training.dataset import load_and_prepare_dataset


@pytest.fixture
def mock_tokenizer():
    """Mock Hugging Face tokenizer that returns Arrow-serializable primitives."""
    tokenizer = MagicMock()

    def fake_tokenize(batch_tokens, is_split_into_words=True, **kwargs):
        input_ids = []
        word_ids_per_batch = []

        for tokens in batch_tokens:
            # Fake subword tokenization: [101] + token_ids + [102]
            ids = [101] + [1000 + i for i in range(len(tokens))] + [102]
            w_ids = [None] + list(range(len(tokens))) + [None]
            input_ids.append(ids)
            word_ids_per_batch.append(w_ids)

        class FakeBatchEncoding(dict):
            def word_ids(self, batch_index=0, *args, **kwargs):
                return word_ids_per_batch[batch_index]

        return FakeBatchEncoding({
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        })

    tokenizer.side_effect = fake_tokenize
    return tokenizer


@pytest.fixture
def raw_dataset_dict():
    return {
        "train": [
            {"tokens": ["armaque", "virum"], "tags": ["TITLE_NONE", "LOWER_COMMA"]},
            {"tokens": ["cano"], "tags": ["LOWER_PERIOD"]},
        ]
    }


def test_load_and_prepare_dataset(mocker, mock_tokenizer, raw_dataset_dict):
    """Test dataset filtering, split generation, and -100 subword label alignment."""
    mock_datasets = mocker.patch("latin_itn_training.dataset.load_dataset")

    from datasets import Dataset, DatasetDict
    ds_train = Dataset.from_list(raw_dataset_dict["train"])
    ds_dict = DatasetDict({"train": ds_train})
    mock_datasets.return_value = ds_dict

    train_ds, test_ds = load_and_prepare_dataset(
        tokenizer=mock_tokenizer,
        dataset_name="dummy/dataset",
        max_length=512,
        seed=42,
    )

    assert len(train_ds) > 0
    assert len(test_ds) > 0

    sample_labels = train_ds[0]["labels"]
    assert sample_labels[0] == -100
    assert sample_labels[-1] == -100


def test_load_and_prepare_dataset_invalid_tag_raises(mocker, mock_tokenizer):
    """Test that invalid dataset tags not present in config schema raise a ValueError."""
    from datasets import Dataset, DatasetDict
    invalid_ds = Dataset.from_list([{"tokens": ["test"], "tags": ["INVALID_TAG_SCHEMA"]}])
    mocker.patch("latin_itn_training.dataset.load_dataset", return_value=DatasetDict({"train": invalid_ds}))

    with pytest.raises(ValueError, match="Dataset contains tags not present in config.TAG_LIST"):
        load_and_prepare_dataset(mock_tokenizer, "dummy/dataset")


def test_subtoken_label_alignment_multibyte_and_hyphens(mocker):
    """Verify label alignment handles multi-byte subwords and hyphenated splits properly."""
    mock_tok = mocker.MagicMock()

    class FakeBatchEncoding(dict):
        def word_ids(self, batch_index=0, *args, **kwargs):
            # Map [CLS], word_0_sub1, word_0_sub2, word_1, [SEP]
            return [None, 0, 0, 1, None]

    def fake_tokenize(batch_tokens, **kwargs):
        input_ids = [[101, 1000, 1001, 1002, 102]] * len(batch_tokens)
        attention_mask = [[1, 1, 1, 1, 1]] * len(batch_tokens)
        return FakeBatchEncoding({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

    mock_tok.side_effect = fake_tokenize

    from datasets import Dataset, DatasetDict

    from latin_itn_training.dataset import load_and_prepare_dataset

    sample = {"tokens": ["armaque", "virum"], "tags": ["TITLE_NONE", "LOWER_NONE"]}
    # Provide 20 samples so train_test_split(test_size=0.05) has enough data
    ds_dict = DatasetDict({"train": Dataset.from_list([sample] * 20)})
    mocker.patch("latin_itn_training.dataset.load_dataset", return_value=ds_dict)

    train_ds, test_ds = load_and_prepare_dataset(
        tokenizer=mock_tok,
        dataset_name="dummy/dataset",
    )

    # Subtoken 0 gets tag ID, subtoken 1 (tail) gets -100
    labels = train_ds[0]["labels"]
    assert labels[1] != -100  # First subtoken gets actual label ID
    assert labels[2] == -100  # Tail subtoken masked out


def test_max_samples_subsampling(mocker, mock_tokenizer):
    """Verify max_samples restricts the dataset size early before mapping."""
    from datasets import Dataset, DatasetDict
    
    samples = [{"tokens": ["cano"], "tags": ["LOWER_PERIOD"]}] * 20
    ds_dict = DatasetDict({"train": Dataset.from_list(samples)})
    mocker.patch("latin_itn_training.dataset.load_dataset", return_value=ds_dict)

    train_ds, test_ds = load_and_prepare_dataset(
        tokenizer=mock_tokenizer,
        dataset_name="dummy/dataset",
        max_samples=10,
    )

    # Total loaded samples should not exceed max_samples (10 split into ~9 train / 1 test)
    assert len(train_ds) + len(test_ds) == 10



def test_sequence_length_filtering(mocker):
    """Verify sequences exceeding max_length subtokens are filtered out."""
    mock_tok = mocker.MagicMock()

    class FakeBatchEncoding(dict):
        def word_ids(self, batch_index=0, *args, **kwargs):
            # 8 total subtokens
            return [None] + list(range(6)) + [None]

    def fake_tokenize(batch_tokens, **kwargs):
        return FakeBatchEncoding({
            "input_ids": [[101] + [1000] * 6 + [102]],
            "attention_mask": [[1] * 8],
        })

    mock_tok.side_effect = fake_tokenize

    from datasets import Dataset, DatasetDict

    from latin_itn_training.dataset import load_and_prepare_dataset

    long_sample = {"tokens": ["a", "b", "c", "d", "e", "f"], "tags": ["LOWER_NONE"] * 6}
    ds_dict = DatasetDict({"train": Dataset.from_list([long_sample])})
    mocker.patch("latin_itn_training.dataset.load_dataset", return_value=ds_dict)

    train_ds, test_ds = load_and_prepare_dataset(
        tokenizer=mock_tok,
        dataset_name="dummy/dataset",
        max_length=5,  # Max length 5 will filter out sequence of 8 subtokens
    )

    assert len(train_ds) == 0
    assert len(test_ds) == 0


def test_existing_validation_split_remapped_to_test(mocker, mock_tokenizer):
    """Verify that when 'validation' exists but 'test' is missing, 'validation' becomes 'test'."""
    from datasets import Dataset, DatasetDict
    
    train_data = [{"tokens": ["armaque"], "tags": ["TITLE_NONE"]}]
    val_data = [{"tokens": ["cano"], "tags": ["LOWER_PERIOD"]}]
    
    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
    })
    mocker.patch("latin_itn_training.dataset.load_dataset", return_value=ds_dict)

    train_ds, test_ds = load_and_prepare_dataset(
        tokenizer=mock_tokenizer,
        dataset_name="dummy/dataset",
    )

    assert len(train_ds) == 1
    assert len(test_ds) == 1


def test_single_dataset_input_fallback(mocker, mock_tokenizer):
    """Verify dataset processing functions correctly when load_dataset returns a raw Dataset (not DatasetDict)."""
    from datasets import Dataset
    
    raw_ds = Dataset.from_list([
        {"tokens": ["armaque", "virum"], "tags": ["TITLE_NONE", "LOWER_COMMA"]},
        {"tokens": ["cano"], "tags": ["LOWER_PERIOD"]},
    ])
    mocker.patch("latin_itn_training.dataset.load_dataset", return_value=raw_ds)

    train_ds, test_ds = load_and_prepare_dataset(
        tokenizer=mock_tokenizer,
        dataset_name="dummy/dataset",
    )

    assert len(train_ds) > 0
    assert len(test_ds) > 0