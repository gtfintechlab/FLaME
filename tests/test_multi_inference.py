import pytest
from main import run_tasks, parse_arguments
import yaml
import sys
from unittest.mock import patch


@pytest.mark.parametrize(
    "tasks",
    [
        ["fomc", "numclaim", "finer"],  # Three tasks that are registered
        ["fomc"],  # Single task
        ["numclaim", "finer"],  # Two tasks
    ],
)
def test_multitask_inference(dummy_args, tasks):
    """
    Smoke-test the multi-task inference runner with various task combinations.
    Dummy fixtures stub I/O, so we only check for no exceptions.
    """
    dummy_args.mode = "inference"
    # ensure we use tasks list not dataset
    dummy_args.dataset = None
    dummy_args.tasks = tasks
    run_tasks(tasks, dummy_args.mode, dummy_args)


@patch("main.supported_tasks")
def test_multitask_with_custom_tasks(mock_supported, dummy_args):
    """
    Test multi-task with custom tasks that are not in the registry.
    """
    # Define our custom tasks including fpb
    custom_tasks = ["fpb", "fomc", "numclaim"]

    # Mock the supported_tasks function to return our custom tasks
    mock_supported.return_value = set(custom_tasks)

    # Set up the args
    dummy_args.mode = "inference"
    dummy_args.dataset = None
    dummy_args.tasks = custom_tasks

    # Mock the inference function to avoid actual execution
    with patch("main.inference"):
        # This should now work because we've mocked supported_tasks
        run_tasks(custom_tasks, dummy_args.mode, dummy_args)


def test_yaml_config_tasks(tmp_path, monkeypatch):
    """
    Test that tasks specified in YAML config are properly read and executed.
    """
    # Create temp YAML with registered tasks
    config = {
        "tasks": ["fomc", "numclaim", "finer"],  # Tasks that are in registry
        "mode": "inference",
        "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Mock the command line args
    argv = ["main.py", "--config", str(config_file)]
    monkeypatch.setattr(sys, "argv", argv)

    # Parse arguments and verify tasks
    args = parse_arguments()
    assert args.tasks == ["fomc", "numclaim", "finer"]
    assert args.mode == "inference"
    assert args.model == "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
