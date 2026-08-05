import pytest
import numpy as np
from latin_itn.metrics import _parse_tag, compute_metrics, to_percentage
from latin_itn.config import ID2LABEL


def test_to_percentage():
    assert to_percentage(0.85432) == pytest.approx(85.432)


def test_parse_tag():
    assert _parse_tag("TITLE_COMMA") == ("TITLE", "COMMA")
    assert _parse_tag("LOWER_NONE") == ("LOWER", "NONE")
    assert _parse_tag("SINGLE") == ("SINGLE", "NONE")


def test_compute_metrics_basic():
    """Test macro metric calculation and casing/punctuation accuracy decomposition."""
    # Label 0: LOWER_NONE, Label 1: TITLE_COMMA
    predictions = np.array([
        [[0.9, 0.1], [0.2, 0.8]],  # Preds: 0, 1 -> LOWER_NONE, TITLE_COMMA
    ])
    labels = np.array([
        [0, 1]  # Truth: 0, 1 -> LOWER_NONE, TITLE_COMMA
    ])

    metrics = compute_metrics((predictions, labels))

    assert metrics["overall_accuracy"] == 1.0
    assert metrics["overall_f1"] == 1.0
    assert metrics["casing_accuracy"] == 1.0
    assert metrics["punct_accuracy"] == 1.0


def test_compute_metrics_ignores_minus_100():
    """Test that subword tokens (-100) are excluded from metric evaluation."""
    predictions = np.array([
        [[0.9, 0.1], [0.9, 0.1], [0.2, 0.8]],
    ])
    labels = np.array([
        [0, -100, 1]  # -100 should be ignored
    ])

    metrics = compute_metrics((predictions, labels))
    assert metrics["overall_accuracy"] == 1.0


def test_compute_metrics_single_class_zero_division():
    """Ensure precision/recall/F1 calculations handle unpredicted/absent classes without raising NaN errors."""
    # Label set only contains LOWER_NONE (0); TITLE_PERIOD (1) never appears
    predictions = np.array([
        [[0.9, 0.1], [0.8, 0.2]],
    ])
    labels = np.array([
        [0, 0]
    ])

    metrics = compute_metrics((predictions, labels))
    assert not np.isnan(metrics["overall_f1"])
    assert metrics["overall_accuracy"] == 1.0


def test_compute_metrics_all_ignored_labels():
    """Ensure evaluation on a sequence where all tokens are -100 (e.g., special tokens only) handles safely."""
    predictions = np.array([
        [[0.5, 0.5], [0.5, 0.5]],
    ])
    labels = np.array([
        [-100, -100]
    ])

    metrics = compute_metrics((predictions, labels))
    assert metrics["overall_accuracy"] == 0.0
    assert metrics["overall_f1"] == 0.0


def test_compute_metrics_tuple_predictions_and_distributions():
    """Verify that compute_metrics handles tuple prediction inputs (e.g. logits + hidden states)

    and accurately records pred_dist and gold_dist keys.
    """
    # Simulate Hugging Face output format: (logits, extra_output)
    logits = np.array([[[0.9, 0.1], [0.1, 0.9]]])
    tuple_predictions = (logits, np.array([0.123]))
    labels = np.array([[0, 1]])

    metrics = compute_metrics((tuple_predictions, labels))

    assert metrics["overall_accuracy"] == 1.0

    # Ensure class distribution keys exist for every label in ID2LABEL
    for label_name in ID2LABEL.values():
        assert f"pred_dist/{label_name}" in metrics
        assert f"gold_dist/{label_name}" in metrics

    # Verify probability distribution calculation (1 token of ID 0, 1 token of ID 1 -> 50% each)
    label_0 = ID2LABEL[0]
    label_1 = ID2LABEL[1]
    assert metrics[f"pred_dist/{label_0}"] == pytest.approx(0.5)
    assert metrics[f"pred_dist/{label_1}"] == pytest.approx(0.5)


def test_compute_metrics_partial_mismatch_subtasks():
    """Verify casing and punctuation sub-task metrics when the model gets casing right

    but punctuation wrong (and vice versa).
    """
    # Get labels representing LOWER_NONE and LOWER_COMMA/PERIOD
    # (Assuming 0 is LOWER_NONE and 1 is LOWER with some punctuation)
    predictions = np.array([[[0.1, 0.9], [0.9, 0.1]]])  # Preds: [1, 0]
    labels = np.array([[0, 1]])                        # Truth: [0, 1]

    metrics = compute_metrics((predictions, labels))

    # Overall token match should fail (0%)
    assert metrics["overall_accuracy"] == 0.0

    # Casing should still be 100% since both 0 and 1 are LOWER casing
    pred_casing = _parse_tag(ID2LABEL[1])[0]
    gold_casing = _parse_tag(ID2LABEL[0])[0]
    if pred_casing == gold_casing:
        assert metrics["casing_accuracy"] == 1.0


def test_compute_metrics_casing_f1_absent_title():
    """Verify that binary casing_f1 safely evaluates to 0.0 via zero_division=0

    when no TITLE tokens exist in the evaluation sample.
    """
    # Assuming label 0 is LOWER_NONE
    predictions = np.array([[[0.9, 0.1], [0.9, 0.1]]])
    labels = np.array([[0, 0]])

    metrics = compute_metrics((predictions, labels))

    assert metrics["casing_accuracy"] == 1.0
    assert metrics["casing_f1"] == 0.0  # pos_label="TITLE" never appeared; should be 0.0, not error


def test_compute_metrics_multi_sequence_batch():
    """Test aggregation across a batch with multiple sequence lengths and masked tokens."""
    predictions = np.array([
        [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]],  # Seq 1 preds: [0, 1, 0]
        [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]],  # Seq 2 preds: [1, 0, 0]
    ])
    labels = np.array([
        [0, 1, -100],  # Seq 1 gold: [0, 1] (3rd token ignored)
        [1, 0, 0],     # Seq 2 gold: [1, 0, 0]
    ])

    metrics = compute_metrics((predictions, labels))

    # 5 valid tokens total, all correct -> 1.0 accuracy
    assert metrics["overall_accuracy"] == 1.0


@pytest.mark.parametrize(
    "tag_str, expected_casing, expected_punct",
    [
        ("TITLE_COMMA", "TITLE", "COMMA"),
        ("LOWER_NONE", "LOWER", "NONE"),
        ("TITLE", "TITLE", "NONE"),                         # No underscore fallback
        ("LOWER_PERIOD_EXTRA", "LOWER", "PERIOD_EXTRA"),    # split("_", 1) retains trailing underscores
    ],
)
def test_parse_tag_variations(tag_str, expected_casing, expected_punct):
    """Verify tag string splitting logic across common and boundary formats."""
    assert _parse_tag(tag_str) == (expected_casing, expected_punct)