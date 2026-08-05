from unittest.mock import MagicMock, patch
import torch

from latin_itn.predictor import (
    LatinITNPredictor,
    _predict_chunk_tags,
    run_strided_inference,
)


def test_predict_chunk_tags():
    """Test chunk tag extraction and alignment with word boundary positions."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Simulate model prediction IDs (e.g., 0 -> LOWER_NONE, 7 -> TITLE_NONE)
    mock_logits = torch.zeros((1, 3, 14))
    mock_logits[0, 1, 7] = 10.0  # Word 0 -> TITLE_NONE
    mock_logits[0, 2, 0] = 10.0  # Word 1 -> LOWER_NONE

    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs

    mock_model.config.id2label = {0: "LOWER_NONE", 7: "TITLE_NONE"}

    # Mock tokenizer outputs
    mock_inputs = MagicMock()
    mock_inputs.word_ids.return_value = [None, 0, 1]
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer.return_value = mock_inputs

    tags = _predict_chunk_tags(["caesar", "dicit"], mock_model, mock_tokenizer, device="cpu")

    assert tags[0] == "TITLE_NONE"
    assert tags[1] == "LOWER_NONE"


@patch("latin_itn.predictor._predict_chunk_tags")
def test_run_strided_inference_short_text(mock_predict_tags):
    """Test single-window execution when input length is within window bounds."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_cltk = MagicMock()

    mock_cltk.tokenize.return_value = [("dicam", False), ("plane", False)]
    mock_predict_tags.return_value = ["TITLE_COMMA", "LOWER_PERIOD"]

    output = run_strided_inference(
        text="dicam plane",
        model=mock_model,
        tokenizer=mock_tokenizer,
        cltk_tok=mock_cltk,
        device="cpu",
        window_tokens=10,
        stride_tokens=5,
    )

    assert output == "Dicam, plane."


@patch("latin_itn.predictor._predict_chunk_tags")
def test_strided_inference_multi_window_overlap(mock_predict_tags):
    """Ensure strided inference correctly processes multi-window streams without dropping tokens."""
    mock_model, mock_tokenizer, mock_cltk = MagicMock(), MagicMock(), MagicMock()

    # 10 words -> forcing multiple windows with window=4, stride=2 (implied margin=1)
    words = [f"word{i}" for i in range(10)]
    mock_cltk.tokenize.return_value = [(w, False) for w in words]

    # Return dummy valid tag for whatever chunk length is passed in
    mock_predict_tags.side_effect = lambda chunk, *args, **kwargs: ["TITLE_NONE"] + ["LOWER_NONE"] * (len(chunk) - 1)

    output = run_strided_inference(
        text=" ".join(words),
        model=mock_model,
        tokenizer=mock_tokenizer,
        cltk_tok=mock_cltk,
        device="cpu",
        window_tokens=4,
        stride_tokens=2,
    )

    # Verify every single word survived reconstruction and formatting
    reconstructed_words = output.replace(".", "").split()
    assert len(reconstructed_words) == 10
    assert reconstructed_words[0] == "Word0"  # Default title casing on first word


@patch("latin_itn.predictor._predict_chunk_tags")
def test_strided_inference_small_window_stride(mock_predict_tags):
    """Verify behavior when stride step size is 1."""
    mock_model, mock_tokenizer, mock_cltk = MagicMock(), MagicMock(), MagicMock()

    words = ["arma", "virum", "cano"]
    mock_cltk.tokenize.return_value = [(w, False) for w in words]
    mock_predict_tags.side_effect = lambda chunk, *args, **kwargs: ["LOWER_NONE"] * len(chunk)

    output = run_strided_inference(
        text="arma virum cano",
        model=mock_model,
        tokenizer=mock_tokenizer,
        cltk_tok=mock_cltk,
        device="cpu",
        window_tokens=2,
        stride_tokens=1,
    )

    reconstructed_words = output.replace(".", "").split()
    assert len(reconstructed_words) == 3


@patch("latin_itn.predictor.run_strided_inference")
def test_latin_itn_predictor_class(mock_run_strided):
    """Test that LatinITNPredictor wrapper delegates correctly to run_strided_inference."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_cltk = MagicMock()
    mock_run_strided.return_value = "Arma virumque cano."

    predictor = LatinITNPredictor(
        model=mock_model,
        tokenizer=mock_tokenizer,
        cltk_tok=mock_cltk,
        device="cpu",
    )

    res = predictor.predict("arma virumque cano", window_tokens=100, stride_tokens=50)

    assert res == "Arma virumque cano."
    mock_run_strided.assert_called_once_with(
        text="arma virumque cano",
        model=mock_model,
        tokenizer=mock_tokenizer,
        cltk_tok=mock_cltk,
        device="cpu",
        window_tokens=100,
        stride_tokens=50,
    )


@patch("latin_itn.predictor.get_model_and_tokenizer")
def test_latin_itn_predictor_from_pretrained(mock_get_model_and_tok):
    """Test LatinITNPredictor.from_pretrained initialization."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_get_model_and_tok.return_value = (mock_model, mock_tokenizer)

    predictor = LatinITNPredictor.from_pretrained("dummy/model-path", device="cpu")

    assert predictor.model == mock_model
    assert predictor.tokenizer == mock_tokenizer
    mock_model.to.assert_called_once_with("cpu")
    mock_model.eval.assert_called_once()


def test_predict_chunk_tags_subword_alignment():
    """Verify _predict_chunk_tags takes prediction from the FIRST subtoken of a split word."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Logits for 5 subtokens: [CLS], arma, ##que, virum, [SEP]
    mock_logits = torch.zeros((1, 5, 3))
    mock_logits[0, 1, 1] = 10.0  # Subtoken 0 ("arma") -> Tag ID 1 ("TITLE_NONE")
    mock_logits[0, 2, 2] = 10.0  # Subtoken 1 ("##que") -> Tag ID 2 ("LOWER_PERIOD") - SHOULD BE IGNORED
    mock_logits[0, 3, 0] = 10.0  # Subtoken 2 ("virum") -> Tag ID 0 ("LOWER_NONE")

    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs
    mock_model.config.id2label = {0: "LOWER_NONE", 1: "TITLE_NONE", 2: "LOWER_PERIOD"}

    mock_inputs = MagicMock()
    mock_inputs.word_ids.return_value = [None, 0, 0, 1, None]
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer.return_value = mock_inputs

    tags = _predict_chunk_tags(["armaque", "virum"], mock_model, mock_tokenizer, device="cpu")

    # Second subtoken "##que" tag (LOWER_PERIOD) must be ignored in favor of first subtoken (TITLE_NONE)
    assert tags == ["TITLE_NONE", "LOWER_NONE"]


@patch("latin_itn.predictor._predict_chunk_tags")
def test_strided_inference_tag_stitching_precision(mock_predict_tags):
    """Verify that strided inference stitches tags from overlapping windows at exact boundaries."""
    mock_model, mock_tokenizer, mock_cltk = MagicMock(), MagicMock(), MagicMock()

    # 6 words: w0, w1, w2, w3, w4, w5
    words = [f"w{i}" for i in range(6)]
    mock_cltk.tokenize.return_value = [(w, False) for w in words]

    # Window 4, Stride 2 -> Margin 1
    # Chunk 0 (w0..w3): Returns ["TITLE_NONE"] * 4
    # Chunk 1 (w2..w5): Returns ["LOWER_PERIOD"] * 4
    def side_effect(chunk_strings, *args, **kwargs):
        if chunk_strings[0] == "w0":
            return ["TITLE_NONE"] * len(chunk_strings)
        return ["LOWER_PERIOD"] * len(chunk_strings)

    mock_predict_tags.side_effect = side_effect

    # We patch rejoin_cltk_enclitics_and_format to return the tags directly for verification
    with patch("latin_itn.predictor.rejoin_cltk_enclitics_and_format", side_effect=lambda meta, tags: tags):
        tags = run_strided_inference(
            text="w0 w1 w2 w3 w4 w5",
            model=mock_model,
            tokenizer=mock_tokenizer,
            cltk_tok=mock_cltk,
            device="cpu",
            window_tokens=4,
            stride_tokens=2,
        )

    # Chunk 0 covers indices [0, 1, 2] (local_end = 4 - 1 = 3) -> TITLE_NONE
    # Chunk 1 covers remaining indices [3, 4, 5] -> LOWER_PERIOD
    assert tags[:3] == ["TITLE_NONE", "TITLE_NONE", "TITLE_NONE"]
    assert tags[3:] == ["LOWER_PERIOD", "LOWER_PERIOD", "LOWER_PERIOD"]


def test_run_strided_inference_empty_text():
    """Verify empty or whitespace-only input safely returns empty string."""
    mock_model, mock_tokenizer, mock_cltk = MagicMock(), MagicMock(), MagicMock()
    mock_cltk.tokenize.return_value = []

    output = run_strided_inference(
        text="   ",
        model=mock_model,
        tokenizer=mock_tokenizer,
        cltk_tok=mock_cltk,
    )

    assert output == ""


def test_predict_chunk_tags_truncation_fallback():
    """Verify padding with LOWER_NONE if model output yields fewer word predictions than input tokens."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Logits return predictions for only 1 token
    mock_logits = torch.zeros((1, 1, 1))
    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs
    mock_model.config.id2label = {0: "LOWER_NONE"}

    mock_inputs = MagicMock()
    mock_inputs.word_ids.return_value = [0]
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer.return_value = mock_inputs

    # Expecting 3 words, but tokenizer/logits only returned 1 word ID
    tags = _predict_chunk_tags(["w0", "w1", "w2"], mock_model, mock_tokenizer, device="cpu")

    assert len(tags) == 3
    assert tags == ["LOWER_NONE", "LOWER_NONE", "LOWER_NONE"]


@patch("latin_itn.predictor.torch.cuda.is_available", return_value=False)
@patch("latin_itn.predictor.get_model_and_tokenizer")
def test_latin_itn_predictor_from_pretrained_default_device(mock_get_model_and_tok, mock_cuda_avail):
    """Test LatinITNPredictor.from_pretrained auto-detects CPU when CUDA is unavailable."""
    mock_model, mock_tokenizer = MagicMock(), MagicMock()
    mock_get_model_and_tok.return_value = (mock_model, mock_tokenizer)

    predictor = LatinITNPredictor.from_pretrained("dummy/model-path")

    assert predictor.device == "cpu"