"""Tests for error handling in multi-task execution."""

import pytest
from types import SimpleNamespace
from main import run_tasks, MultiTaskError
from unittest.mock import patch
import pandas as pd

pytestmark = pytest.mark.unit


def test_multitask_error_class():
    """Test that MultiTaskError correctly stores errors."""
    # Create a MultiTaskError with a dict of task errors
    task_errors = {
        "task1": ValueError("Error in task1"),
        "task2": RuntimeError("Error in task2"),
    }
    error = MultiTaskError(task_errors)

    # Error message should include task names
    assert "task1" in str(error)
    assert "task2" in str(error)

    # Error object should store the original errors
    assert error.errors == task_errors
    assert isinstance(error.errors["task1"], ValueError)
    assert isinstance(error.errors["task2"], RuntimeError)


@pytest.mark.parametrize("mode", ["inference", "evaluate"])
def test_run_tasks_collects_errors(dummy_args, mode):
    """Test that run_tasks collects errors from multiple tasks in both modes."""
    # Set up dummy args for the specified mode
    dummy_args.mode = mode
    if mode == "evaluate":
        dummy_args.file_name = "dummy_file.csv"

    # Define mock tasks that will succeed and fail
    mock_tasks = ["success_task", "fail_task1", "fail_task2", "success_task2"]

    # Mock the supported tasks to include our mock tasks
    with patch("main.supported_tasks", return_value=set(mock_tasks)):
        # Mock function to succeed or fail based on task name
        def mock_function(args):
            if args.task.startswith("fail"):
                raise ValueError(f"Error in {args.task}")
            return {"result": "success"}

        # Patch the appropriate function based on mode
        target = "main.inference" if mode == "inference" else "main.evaluate"

        # Run the multi-task execution with patched functions
        with patch(target, side_effect=mock_function):
            # Should raise MultiTaskError with collected errors
            with pytest.raises(MultiTaskError) as excinfo:
                run_tasks(mock_tasks, mode, dummy_args)

            # Check that the errors were collected correctly
            errors = excinfo.value.errors
            assert len(errors) == 2
            assert "fail_task1" in errors
            assert "fail_task2" in errors
            assert "success_task" not in errors
            assert "success_task2" not in errors

            # Check error types
            assert isinstance(errors["fail_task1"], ValueError)
            assert isinstance(errors["fail_task2"], ValueError)


@pytest.mark.parametrize("mode", ["inference", "evaluate"])
def test_run_tasks_continues_after_error(dummy_args, mode):
    """Test that run_tasks continues running tasks after errors in both modes."""
    dummy_args.mode = mode
    if mode == "evaluate":
        dummy_args.file_name = "dummy_file.csv"

    # Track execution order
    executed_tasks = []

    def mock_function(args):
        executed_tasks.append(args.task)
        if args.task == "fail_task":
            raise ValueError("Planned failure")
        return {"result": "success"}

    mock_tasks = ["task1", "fail_task", "task3"]

    # Mock the supported tasks function
    with patch("main.supported_tasks", return_value=set(mock_tasks)):
        # Run with the mock function
        target = "main.inference" if mode == "inference" else "main.evaluate"
        with patch(target, side_effect=mock_function):
            # Should raise MultiTaskError
            with pytest.raises(MultiTaskError):
                run_tasks(mock_tasks, mode, dummy_args)

            # All tasks should have been executed
            assert executed_tasks == ["task1", "fail_task", "task3"]


def test_multitask_error_aggregation_detailed():
    """Test that MultiTaskError correctly aggregates failures with details"""
    # Keep track of calls
    calls = []

    # Create a mock that fails for specific tasks
    def mock_inference(args):
        calls.append(args.task)
        if args.task == "numclaim":
            raise ValueError("Simulated numclaim error")
        elif args.task == "finer":
            raise RuntimeError("Simulated finer error")
        # fomc succeeds
        return pd.DataFrame({"result": ["success"]})

    # Create more complete args to avoid attribute errors
    args = SimpleNamespace(
        tasks=["fomc", "numclaim", "finer"],
        mode="inference",
        model="test-model",
        prompt_format="zero_shot",
        batch_size=10,
        max_tokens=128,
        temperature=0.0,
        top_p=0.9,
        top_k=None,
        repetition_penalty=1.0,
    )

    # Mock the inference function at the main level
    with patch("main.inference", mock_inference):
        # Run tasks and expect MultiTaskError
        with pytest.raises(MultiTaskError) as exc_info:
            run_tasks(args.tasks, args.mode, args)

        # Verify all tasks were attempted
        assert calls == ["fomc", "numclaim", "finer"]

        # Check that the error contains failures for the right tasks
        error = exc_info.value
        assert "numclaim" in error.errors
        assert "finer" in error.errors
        assert "fomc" not in error.errors  # This one succeeded
        assert isinstance(error.errors["numclaim"], ValueError)
        assert isinstance(error.errors["finer"], RuntimeError)


def test_error_recovery_and_reporting():
    """Test error recovery and reporting in multi-task execution"""
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
