from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from latin_itn.config import TrainingConfig
from latin_itn.trainer_utils import (
    ModalProgressLogger,
    build_training_args,
    execute_training_step,
    parse_emissions_summary,
    publish_model_and_readme,
)


@pytest.fixture
def mock_training_config():
    return TrainingConfig(
        hf_repo_id="user/latin-itn",
        base_model="latincy/latin-bert",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        bf16=False,
        fp16=False,
    )


# ---------------------------------------------------------------------------
# Tests for ModalProgressLogger
# ---------------------------------------------------------------------------

def test_logger_format_time():
    logger = ModalProgressLogger()
    assert logger._format_time(45) == "00m 45s"
    assert logger._format_time(125) == "02m 05s"
    assert logger._format_time(3665) == "1h 01m 05s"


def test_logger_render_bar():
    logger = ModalProgressLogger(bar_width=10)
    bar_0 = logger._render_bar(0, 100)
    assert "█" not in bar_0

    bar_100 = logger._render_bar(100, 100)
    assert "█" * 10 in bar_100


def test_logger_on_log_output(capsys):
    logger = ModalProgressLogger(bar_width=10)
    logger.start_time = 100.0

    mock_state = MagicMock()
    mock_state.is_world_process_zero = True
    mock_state.max_steps = 100
    mock_state.global_step = 50
    mock_state.epoch = 1.0

    logs = {"loss": 0.4521}

    with patch("time.time", return_value=200.0):
        logger.on_log(args=MagicMock(), state=mock_state, control=MagicMock(), logs=logs)

    captured = capsys.readouterr()
    assert "50.0%" in captured.out
    assert "Ep" in captured.out
    assert "1.00" in captured.out
    assert "0.4521" in captured.out


def test_logger_on_evaluate_output(capsys):
    logger = ModalProgressLogger()
    mock_state = MagicMock()
    mock_state.is_world_process_zero = True

    metrics = {
        "epoch": 2.0,
        "eval_loss": 0.3500,
        "eval_overall_f1": 0.85,
        "eval_overall_accuracy": 0.90,
        "eval_casing_accuracy": 0.95,
        "eval_punct_accuracy": 0.92,
    }

    logger.on_evaluate(args=MagicMock(), state=mock_state, control=MagicMock(), metrics=metrics)
    captured = capsys.readouterr()

    assert "[Eval Epoch 2.0]" in captured.out
    assert "Val Loss:" in captured.out
    assert "85.00%" in captured.out


# ---------------------------------------------------------------------------
# Tests for build_training_args
# ---------------------------------------------------------------------------

@patch("latin_itn.trainer_utils.TrainingArguments")
def test_build_training_args(mock_training_args, mock_training_config):
    build_training_args(
        output_dir="/tmp/test_out",
        config=mock_training_config,
        num_epochs=3,
        branch_name="main",
        hf_token="fake_token",
        run_name="test_run",
        is_eval_enabled=True,
    )

    mock_training_args.assert_called_once()
    _, kwargs = mock_training_args.call_args
    assert kwargs["eval_strategy"] == "epoch"
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "overall_f1"


# ---------------------------------------------------------------------------
# Tests for publish_model_and_readme
# ---------------------------------------------------------------------------

def test_publish_model_and_readme(tmp_path):
    mock_trainer = MagicMock()
    mock_api = MagicMock()

    out_dir = str(tmp_path)
    readme_content = "# Latin ITN Model"

    publish_model_and_readme(
        trainer=mock_trainer,
        api=mock_api,
        readme_content=readme_content,
        output_dir=out_dir,
        hf_repo_id="user/latin-itn",
        branch_name="main",
        commit_msg="Update ITN model",
    )

    mock_trainer.save_model.assert_called_once_with(out_dir)

    written_readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert written_readme == readme_content

    mock_api.upload_folder.assert_called_once_with(
        folder_path=out_dir,
        repo_id="user/latin-itn",
        revision="main",
        commit_message="Update ITN model",
        ignore_patterns=["checkpoint-*"],
    )


# ---------------------------------------------------------------------------
# Tests for parse_emissions_summary
# ---------------------------------------------------------------------------

def test_parse_emissions_summary_missing():
    hours, carbon = parse_emissions_summary("non_existent.csv")
    assert hours == "N/A"
    assert carbon == "N/A"


def test_parse_emissions_summary_valid(tmp_path):
    csv_file = tmp_path / "emissions.csv"
    df = pd.DataFrame({"duration": [3600.0, 1800.0], "emissions": [0.025, 0.015]})
    df.to_csv(csv_file, index=False)

    hours, carbon = parse_emissions_summary(str(csv_file))
    assert hours == "1.50 hours"
    assert carbon == "0.0400 kg CO2eq"


# ---------------------------------------------------------------------------
# Tests for execute_training_step
# ---------------------------------------------------------------------------

@patch("wandb.init")
@patch("wandb.finish")
@patch("latin_itn.trainer_utils.get_branch_progress")
@patch("latin_itn.trainer_utils.ensure_branch_exists")
def test_execute_training_step_short_circuit(
    mock_ensure_branch,
    mock_get_progress,
    mock_wandb_finish,
    mock_wandb_init,
    mock_training_config,
):
    mock_api = MagicMock()

    # Simulate completed run (5 completed out of 5 target epochs)
    mock_get_progress.return_value = (5.0, 4.0, 88.5, 90.0, 92.0, 89.0)

    result = execute_training_step(
        config=mock_training_config,
        branch_name="main",
        run_name="itn_run",
        target_epochs=5,
        output_dir="/tmp/out",
        train_ds={"labels": [[0, 1, -100]]},
        eval_ds=MagicMock(),
        tokenizer=MagicMock(),
        compute_metrics=MagicMock(),
        callbacks=[],
        api=mock_api,
        hf_token="fake_token",
        readme_content_fn=lambda t, e: "readme",
        commit_message="Finished",
    )

    assert result["completed_previously"] is True
    assert result["best_epoch"] == 4.0
    assert result["f1"] == 88.5


@patch("wandb.init")
@patch("wandb.finish")
@patch("latin_itn.trainer_utils.get_latest_checkpoint", return_value=None)
@patch("latin_itn.trainer_utils.get_branch_progress", return_value=(0.0, None, 0.0, 0.0, 0.0, 0.0))
@patch("latin_itn.trainer_utils.ensure_branch_exists")
@patch("latin_itn.trainer_utils.get_model_and_tokenizer")
@patch("latin_itn.trainer_utils.WeightedITNTrainer")
@patch("latin_itn.trainer_utils.publish_model_and_readme")
def test_execute_training_step_full_run(
    mock_publish,
    mock_trainer_cls,
    mock_get_model_tok,
    mock_ensure_branch,
    mock_get_progress,
    mock_get_ckpt,
    mock_wandb_finish,
    mock_wandb_init,
    mock_training_config,
):
    mock_api = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_get_model_tok.return_value = (mock_model, mock_tokenizer)

    mock_trainer_instance = MagicMock()
    mock_trainer_cls.return_value = mock_trainer_instance

    mock_trainer_instance.state.log_history = [
        {
            "epoch": 1.0,
            "eval_loss": 0.3,
            "eval_overall_f1": 0.85,
            "eval_overall_accuracy": 0.90,
            "eval_casing_accuracy": 0.95,
            "eval_punct_accuracy": 0.92,
        }
    ]

    result = execute_training_step(
        config=mock_training_config,
        branch_name="main",
        run_name="itn_run",
        target_epochs=1,
        output_dir="/tmp/out",
        train_ds={"labels": [[0, 1, -100]]},
        eval_ds=MagicMock(),
        tokenizer=mock_tokenizer,
        compute_metrics=MagicMock(),
        callbacks=[],
        api=mock_api,
        hf_token="fake_token",
        readme_content_fn=lambda t, e: "generated readme",
        commit_message="Finished ITN run",
    )

    mock_trainer_instance.train.assert_called_once_with(resume_from_checkpoint=None)
    mock_publish.assert_called_once()

    assert result["completed_previously"] is False
    assert result["f1"] == 85.0
    assert result["acc"] == 90.0
    assert result["casing_acc"] == 95.0
    assert result["punct_acc"] == 92.0


def test_execute_training_step_empty_log_history_fallback(mocker, mock_training_config):
    """Ensure training pipeline completes gracefully if log_history contains no evaluation steps."""
    mock_trainer = MagicMock()
    mock_trainer.state.log_history = []  # No evaluation logs recorded!

    mocker.patch("wandb.init")
    mocker.patch("wandb.finish")
    mocker.patch("latin_itn.trainer_utils.WeightedITNTrainer", return_value=mock_trainer)
    mocker.patch("latin_itn.trainer_utils.get_model_and_tokenizer", return_value=(MagicMock(), MagicMock()))
    mocker.patch("latin_itn.trainer_utils.get_branch_progress", return_value=(0.0, None, 0.0, 0.0, 0.0, 0.0))
    mocker.patch("latin_itn.trainer_utils.ensure_branch_exists")
    mocker.patch("latin_itn.trainer_utils.publish_model_and_readme")

    result = execute_training_step(
        config=mock_training_config,
        branch_name="main",
        run_name="test",
        target_epochs=1,
        output_dir="/tmp/out",
        train_ds={"labels": [[0, 1, -100]]},
        eval_ds=MagicMock(),
        tokenizer=MagicMock(),
        compute_metrics=MagicMock(),
        callbacks=[],
        api=MagicMock(),
        hf_token="fake",
        readme_content_fn=lambda t, e: "",
        commit_message="",
    )
    assert result["f1"] == 0.0