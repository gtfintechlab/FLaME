"""Tests for task validation and registry functionality"""

from types import SimpleNamespace

import pytest

from flame.task_registry import EVALUATE_MAP, INFERENCE_MAP, supported
from main import run_tasks

pytestmark = pytest.mark.unit


def test_task_registry_completeness():
    """Test that task registry contains expected tasks"""
    # Get supported tasks
    inference_tasks = supported("inference")
    evaluate_tasks = supported("evaluate")

    # Check that key tasks are present for inference
    expected_inference = {
        "fomc",
        "numclaim",
        "finer",
        "finentity",
        "causal_classification",
        "subjectiveqa",
        "ectsum",
        "fnxl",
    }
    assert expected_inference.issubset(inference_tasks)

    # Check that key tasks are present for evaluation
    expected_evaluation = {
        "numclaim",
        "finer",
        "finentity",
        "fnxl",
        "causal_classification",
        "subjectiveqa",
        "ectsum",
        "refind",
        "banking77",
        "convfinqa",
        "finqa",
        "tatqa",
        "causal_detection",
    }
    assert expected_evaluation.issubset(evaluate_tasks)


def test_invalid_mode_raises_error():
    """Test that invalid mode raises error"""
    with pytest.raises(ValueError, match="mode must be 'inference' or 'evaluate'"):
        supported("invalid_mode")


def test_task_validation_in_run_tasks():
    """Test task validation in run_tasks function"""
    args = SimpleNamespace(
        tasks=["fomc", "invalid_task"], mode="inference", model="test-model"
    )

    with pytest.raises(ValueError, match="Task 'invalid_task' not supported"):
        run_tasks(args.tasks, args.mode, args)


def test_valid_tasks_in_registry():
    """Test that registry tasks are properly validated"""
    # Get some valid tasks from the registry
    valid_inference_tasks = list(supported("inference"))[:3]

    # Verify all tasks are in the inference registry
    assert all(task in supported("inference") for task in valid_inference_tasks)

    # Verify we can create args with these tasks without error
    args = SimpleNamespace(
        tasks=valid_inference_tasks, mode="inference", model="test-model"
    )

    # Check task validation logic specifically
    inference_supported = supported("inference")
    for task in args.tasks:
        assert task in inference_supported


def test_inference_and_evaluation_task_mapping():
    """Test that tasks are correctly mapped in both inference and evaluation maps"""
    # Check that maps are not empty
    assert len(INFERENCE_MAP) > 0
    assert len(EVALUATE_MAP) > 0

    # Check that all keys in maps are strings
    assert all(isinstance(task, str) for task in INFERENCE_MAP.keys())
    assert all(isinstance(task, str) for task in EVALUATE_MAP.keys())

    # Check that all values are callables
    assert all(callable(func) for func in INFERENCE_MAP.values())
    assert all(callable(func) for func in EVALUATE_MAP.values())

    # Verify some tasks are in both maps (like numclaim)
    common_tasks = set(INFERENCE_MAP.keys()) & set(EVALUATE_MAP.keys())
    assert "numclaim" in common_tasks
    assert "finer" in common_tasks

    # Verify some tasks are only in inference (like econlogicqa)
    inference_only = set(INFERENCE_MAP.keys()) - set(EVALUATE_MAP.keys())
    assert "econlogicqa" in inference_only
    assert "finred" in inference_only


def test_supported_function_behavior():
    """Test the supported() function behavior"""
    # Test inference mode
    inference_tasks = supported("inference")
    assert isinstance(inference_tasks, set)
    assert len(inference_tasks) > 0
    assert all(isinstance(task, str) for task in inference_tasks)

    # Test evaluate mode
    evaluate_tasks = supported("evaluate")
    assert isinstance(evaluate_tasks, set)
    assert len(evaluate_tasks) > 0
    assert all(isinstance(task, str) for task in evaluate_tasks)

    # Test case sensitivity
    assert supported("INFERENCE") == supported("inference")
    assert supported("EVALUATE") == supported("evaluate")

    # Test invalid mode
    with pytest.raises(ValueError):
        supported("unknown_mode")


def test_run_tasks_with_nonexistent_task():
    """Test that run_tasks fails with non-existent task"""
    # Create args with a mix of valid and invalid tasks
    args = SimpleNamespace(
        tasks=["fomc", "nonexistent_task", "numclaim"],
        mode="inference",
        model="test-model",
    )

    # Should fail when running tasks
    with pytest.raises(ValueError, match="Task 'nonexistent_task' not supported"):
        run_tasks(args.tasks, args.mode, args)


def test_task_case_sensitivity():
    """Test that task names are case-sensitive"""
    # Verify task names are lowercase in registry
    assert "fomc" in supported("inference")
    assert "FOMC" not in supported("inference")

    # Test validation with incorrect case
    args = SimpleNamespace(
        tasks=["FOMC"],  # Uppercase
        mode="inference",
    )

    with pytest.raises(ValueError, match="Task 'FOMC' not supported"):
        run_tasks(args.tasks, args.mode, args)


def test_empty_task_list_behavior():
    """Test behavior with empty task list"""
    args = SimpleNamespace(tasks=[], mode="inference")

    # Empty task list should run without error (does nothing)
    run_tasks(args.tasks, args.mode, args)
