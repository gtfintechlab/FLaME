"""Tests for task validation functionality."""

import pytest
from unittest.mock import patch
from flame.task_registry import supported, INFERENCE_MAP, EVALUATE_MAP
from main import run_tasks


def test_supported_tasks_inference():
    """Test the supported tasks function returns expected tasks for inference mode."""
    inference_tasks = supported("inference")
    assert isinstance(inference_tasks, set)
    assert len(inference_tasks) > 0
    # Should match the keys in INFERENCE_MAP
    assert inference_tasks == set(INFERENCE_MAP.keys())


def test_supported_tasks_evaluate():
    """Test the supported tasks function returns expected tasks for evaluate mode."""
    evaluate_tasks = supported("evaluate")
    assert isinstance(evaluate_tasks, set)
    assert len(evaluate_tasks) > 0
    # Should match the keys in EVALUATE_MAP
    assert evaluate_tasks == set(EVALUATE_MAP.keys())


def test_supported_tasks_invalid_mode():
    """Test the supported tasks function raises an error for invalid mode."""
    with pytest.raises(ValueError):
        supported("invalid_mode")


def test_supported_tasks_case_insensitive():
    """Test the supported tasks function is case-insensitive for mode."""
    inference_tasks = supported("inference")
    inference_tasks_upper = supported("INFERENCE")
    assert inference_tasks == inference_tasks_upper


def test_run_tasks_validates_tasks_inference(dummy_args):
    """Test that run_tasks validates task names for inference mode."""
    mode = "inference"
    dummy_args.mode = mode

    # Test with valid tasks (should not raise)
    valid_tasks = list(supported(mode))[:2]  # Get first two valid tasks
    run_tasks(valid_tasks, mode, dummy_args)

    # Test with invalid task (should raise ValueError)
    invalid_tasks = ["not_a_real_task"]
    with pytest.raises(ValueError, match=r"Task 'not_a_real_task' not supported"):
        run_tasks(invalid_tasks, mode, dummy_args)

    # Test with mixed valid and invalid tasks (should raise on first invalid)
    mixed_tasks = valid_tasks + ["not_a_real_task"]
    with pytest.raises(ValueError, match=r"Task 'not_a_real_task' not supported"):
        run_tasks(mixed_tasks, mode, dummy_args)


@patch("main.evaluate")
def test_run_tasks_validates_tasks_evaluate(mock_evaluate, dummy_args):
    """Test that run_tasks validates task names for evaluate mode."""
    # Configure the mock to return a tuple of dataframes (as expected by evaluate)
    import pandas as pd

    mock_evaluate.return_value = (pd.DataFrame(), pd.DataFrame())

    mode = "evaluate"
    dummy_args.mode = mode
    dummy_args.file_name = "dummy_file.csv"  # Add a dummy file name

    # Test with valid tasks (should not raise)
    valid_tasks = list(supported(mode))[:2]  # Get first two valid tasks

    # Run with the mocked evaluation function
    run_tasks(valid_tasks, mode, dummy_args)

    # Test with invalid task (should raise ValueError)
    invalid_tasks = ["not_a_real_task"]
    with pytest.raises(ValueError, match=r"Task 'not_a_real_task' not supported"):
        run_tasks(invalid_tasks, mode, dummy_args)

    # Test with mixed valid and invalid tasks (should raise on first invalid)
    mixed_tasks = valid_tasks + ["not_a_real_task"]
    with pytest.raises(ValueError, match=r"Task 'not_a_real_task' not supported"):
        run_tasks(mixed_tasks, mode, dummy_args)
