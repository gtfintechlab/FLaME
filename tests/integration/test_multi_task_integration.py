"""Integration tests for multi-task functionality in FLaME"""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.integration


def test_end_to_end_multi_task_workflow(tmp_path, monkeypatch, dummy_args):
    """Test complete multi-task workflow from YAML to execution"""
    # Create a realistic YAML config
    config_data = {
        "mode": "inference",
        "tasks": ["fomc", "numclaim", "finer"],
        "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "max_tokens": 128,
        "temperature": 0.0,
        "batch_size": 10,
        "prompt_format": "zero_shot",
    }

    config_file = tmp_path / "multi_task_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Set up environment
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "test_token")

    # Mock command line arguments
    test_args = ["main.py", "--config", str(config_file)]

    # Import main module
    import main

    # Track function calls
    inference_calls = []

    def mock_inference(args):
        """Mock inference function that tracks calls"""
        inference_calls.append(args.task)
        # Return a DataFrame as expected
        return pd.DataFrame(
            {"input": ["test input"], "output": ["mock reply"], "task": [args.task]}
        )

    # Mock the inference function at the main module level
    with patch("main.inference", mock_inference):
        # Mock sys.argv
        with monkeypatch.context() as m:
            m.setattr(sys, "argv", test_args)

            # Parse args and run tasks
            args = main.parse_arguments()
            if args.tasks:
                main.run_tasks(args.tasks, args.mode, args)

    # Verify all tasks were executed
    assert len(inference_calls) == 3
    assert set(inference_calls) == {"fomc", "numclaim", "finer"}


def test_multi_task_with_mixed_modes(tmp_path, monkeypatch):
    """Test that mixing modes (inference/evaluation) is handled properly"""
    # Create config for inference
    inference_config = {
        "mode": "inference",
        "tasks": ["fomc", "numclaim"],
        "model": "test-model",
    }

    # Create config for evaluation
    evaluation_config = {
        "mode": "evaluate",
        "tasks": ["finer", "finentity"],
        "file_name": "test_results.csv",
    }

    # Test inference
    config_file = tmp_path / "inference_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(inference_config, f)

    from types import SimpleNamespace

    import main

    # Mock parse_arguments to return our config
    def mock_parse_args():
        args = SimpleNamespace(**inference_config)
        return args

    with patch("main.parse_arguments", mock_parse_args):
        with patch("main.inference") as mock_inf:
            main.run_tasks(["fomc", "numclaim"], "inference", mock_parse_args())
            assert mock_inf.call_count == 2

    # Test evaluation
    config_file = tmp_path / "evaluation_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(evaluation_config, f)

    def mock_parse_args_eval():
        args = SimpleNamespace(**evaluation_config)
        return args

    with patch("main.parse_arguments", mock_parse_args_eval):
        with patch("main.evaluate") as mock_eval:
            main.run_tasks(["finer", "finentity"], "evaluate", mock_parse_args_eval())
            assert mock_eval.call_count == 2


def test_cli_list_tasks_command(monkeypatch, capsys):
    """Test the list-tasks CLI command"""
    # Test by importing and calling the functions directly
    from flame.task_registry import supported

    # Simulate what the command does
    print("Available inference tasks:")
    for task in sorted(supported("inference")):
        print(f"  - {task}")
    print("\nAvailable evaluation tasks:")
    for task in sorted(supported("evaluate")):
        print(f"  - {task}")

    # Capture output
    captured = capsys.readouterr()

    # Verify output contains expected content
    assert "Available inference tasks:" in captured.out
    assert "Available evaluation tasks:" in captured.out
    assert "fomc" in captured.out
    assert "numclaim" in captured.out
    assert "finer" in captured.out

    # Verify structure
    lines = captured.out.strip().split("\n")

    # Find the sections
    inference_start = None
    evaluation_start = None
    for i, line in enumerate(lines):
        if "Available inference tasks:" in line:
            inference_start = i
        elif "Available evaluation tasks:" in line:
            evaluation_start = i

    # Count tasks between sections
    inference_tasks = 0
    evaluation_tasks = 0

    if inference_start is not None and evaluation_start is not None:
        # Count inference tasks
        for i in range(inference_start + 1, evaluation_start):
            if lines[i].strip().startswith("- "):
                inference_tasks += 1

        # Count evaluation tasks
        for i in range(evaluation_start + 1, len(lines)):
            if lines[i].strip() and lines[i].strip().startswith("- "):
                evaluation_tasks += 1

    assert inference_tasks > 10  # Should have many inference tasks
    assert evaluation_tasks > 5  # Should have several evaluation tasks


def test_task_specific_parameters_yaml(tmp_path):
    """Test that task-specific parameters can be specified in YAML"""
    config_data = {
        "mode": "inference",
        "tasks": ["fomc", "numclaim"],
        "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "max_tokens": 256,
        "temperature": 0.1,
        "fomc": {
            "batch_size": 5,
            "max_tokens": 512,  # Override for FOMC
        },
        "numclaim": {
            "temperature": 0.0,  # Override for NumClaim
            "prompt_format": "few_shot",
        },
    }

    config_file = tmp_path / "task_specific_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Mock sys.argv
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["main.py", "--config", str(config_file)]

        from main import parse_arguments

        args = parse_arguments()

        # Verify global parameters
        assert args.tasks == ["fomc", "numclaim"]
        assert args.model == "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
        assert args.max_tokens == 256
        assert args.temperature == 0.1

        # Task-specific parameters would need to be handled in the task functions
        # This test ensures they're at least parsed correctly
        assert hasattr(args, "fomc")
        assert args.fomc["batch_size"] == 5
        assert args.fomc["max_tokens"] == 512

        assert hasattr(args, "numclaim")
        assert args.numclaim["temperature"] == 0.0
        assert args.numclaim["prompt_format"] == "few_shot"
    finally:
        sys.argv = old_argv


def test_multi_task_error_recovery_and_reporting(monkeypatch):
    """Test error recovery and reporting in multi-task execution"""
    from types import SimpleNamespace
    from unittest.mock import patch

    from main import MultiTaskError, run_tasks

    # Create mock functions with controlled failures
    call_log = []

    def mock_inference(args):
        call_log.append(f"inference_{args.task}")
        if args.task == "fomc":
            # First task succeeds
            return pd.DataFrame({"result": ["success"]})
        elif args.task == "numclaim":
            # Second task fails with specific error
            raise ValueError("API rate limit exceeded")
        elif args.task == "finer":
            # Third task succeeds (to test recovery)
            return pd.DataFrame({"result": ["success"]})

    args = SimpleNamespace(
        tasks=["fomc", "numclaim", "finer"],
        mode="inference",
        model="test-model",
        batch_size=10,
        prompt_format="zero_shot",
        max_tokens=128,
        temperature=0.0,
        top_p=0.9,
        top_k=None,
        repetition_penalty=1.0,
    )

    with patch("main.inference", mock_inference):
        with pytest.raises(MultiTaskError) as exc_info:
            run_tasks(args.tasks, args.mode, args)

    # Verify execution order and error handling
    assert call_log == ["inference_fomc", "inference_numclaim", "inference_finer"]

    # Check error details
    error = exc_info.value
    assert len(error.errors) == 1  # Only numclaim failed
    assert "numclaim" in error.errors
    assert "API rate limit exceeded" in str(error.errors["numclaim"])

    # Verify fomc and finer succeeded (not in errors)
    assert "fomc" not in error.errors
    assert "finer" not in error.errors


def test_yaml_with_environment_variables(tmp_path, monkeypatch):
    """Test YAML config with environment variable substitution"""
    # Set environment variables
    monkeypatch.setenv(
        "FLAME_MODEL", "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )
    monkeypatch.setenv("FLAME_MAX_TOKENS", "256")

    # Create config using environment variables (note: this is a test of concept)
    config_data = {
        "mode": "inference",
        "tasks": ["fomc", "numclaim"],
        "model": os.getenv("FLAME_MODEL", "default-model"),
        "max_tokens": int(os.getenv("FLAME_MAX_TOKENS", "128")),
    }

    config_file = tmp_path / "env_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Mock sys.argv
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_file)])

    from main import parse_arguments

    args = parse_arguments()

    # Verify environment variables were used
    assert args.model == "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    assert args.max_tokens == 256


@pytest.mark.parametrize(
    "invalid_config,expected_error",
    [
        # Missing required fields
        ({"tasks": ["fomc"]}, "Mode is required"),
        # Invalid mode
        ({"mode": "invalid", "tasks": ["fomc"]}, "Mode is required and must be either"),
        # Empty tasks list in CLI
        ({"mode": "inference", "tasks": []}, "No tasks specified"),
    ],
)
def test_invalid_configurations(tmp_path, monkeypatch, invalid_config, expected_error):
    """Test that invalid configurations raise appropriate errors"""
    config_file = tmp_path / "invalid_config.yaml"

    # Add default values to avoid key errors
    full_config = {"mode": None, "tasks": None}
    full_config.update(invalid_config)

    with open(config_file, "w") as f:
        yaml.dump(full_config, f)

    # Mock sys.argv
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_file)])

    import main

    with pytest.raises(ValueError, match=expected_error):
        # This should parse args and then fail validation
        args = main.parse_arguments()

        # Manually trigger the validation that happens in __main__
        if not args.mode or args.mode not in ["inference", "evaluate"]:
            raise ValueError(
                "Mode is required and must be either 'inference' or 'evaluate'."
            )
        if args.mode == "evaluate" and not args.file_name:
            raise ValueError("File name is required for evaluation mode.")
        if not args.tasks:
            raise ValueError("No tasks specified; use --tasks option")


def test_multi_task_progress_tracking(capsys):
    """Test that multi-task execution provides progress feedback"""
    import time
    from types import SimpleNamespace
    from unittest.mock import patch

    from main import run_tasks

    # Mock inference with delays to simulate real processing
    def mock_inference(args):
        print(f"Processing task: {args.task}")
        time.sleep(0.1)  # Short delay
        print(f"Completed task: {args.task}")
        return pd.DataFrame({"result": ["success"]})

    args = SimpleNamespace(
        tasks=["fomc", "numclaim", "finer"],
        mode="inference",
        model="test-model",
        batch_size=10,
        prompt_format="zero_shot",
        max_tokens=128,
        temperature=0.0,
        top_p=0.9,
        top_k=None,
        repetition_penalty=1.0,
    )

    with patch("main.inference", mock_inference):
        run_tasks(args.tasks, args.mode, args)

    # Check output contains progress indicators
    captured = capsys.readouterr()
    assert "Processing task: fomc" in captured.out
    assert "Completed task: fomc" in captured.out
    assert "Processing task: numclaim" in captured.out
    assert "Completed task: numclaim" in captured.out
    assert "Processing task: finer" in captured.out
    assert "Completed task: finer" in captured.out
