import json
from unittest.mock import MagicMock, patch
from latin_itn.hf_utils import ensure_branch_exists, get_branch_progress, get_latest_checkpoint


def test_ensure_branch_exists():
    mock_api = MagicMock()
    ensure_branch_exists(mock_api, "user/repo", "main")
    mock_api.create_branch.assert_called_once_with(repo_id="user/repo", branch="main", exist_ok=True)


def test_get_latest_checkpoint(tmp_path):
    # Create fake checkpoint subdirectories
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-200").mkdir()
    (tmp_path / "other-dir").mkdir()

    latest = get_latest_checkpoint(str(tmp_path))
    assert latest.endswith("checkpoint-500")


def test_get_latest_checkpoint_none(tmp_path):
    assert get_latest_checkpoint(str(tmp_path)) is None


@patch("huggingface_hub.hf_hub_download")
def test_get_branch_progress(mock_download, tmp_path):
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = ["trainer_state.json"]

    state_data = {
        "epoch": 2.0,
        "log_history": [
            {
                "epoch": 1.0,
                "eval_loss": 0.4,
                "eval_overall_f1": 0.80,
                "eval_overall_accuracy": 0.85,
                "eval_casing_accuracy": 0.90,
                "eval_punct_accuracy": 0.88,
            }
        ],
    }

    state_file = tmp_path / "trainer_state.json"
    state_file.write_text(json.dumps(state_data), encoding="utf-8")
    mock_download.return_value = str(state_file)

    completed_epochs, best_epoch, best_f1, best_acc, best_casing, best_punct = get_branch_progress(
        mock_api, "user/repo", "main", "fake_token"
    )

    assert completed_epochs == 2.0
    assert best_epoch == 1.0
    assert best_f1 == 80.0
    assert best_acc == 85.0