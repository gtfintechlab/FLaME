"""
Consolidated test suite for prompt functionality.

Tests the prompt registry system that enables task-specific prompt generation
for the FLaME framework.
"""

import pytest

from flame.code.prompts import (
    PromptFormat,
    get_prompt,
    get_prompt_by_name,
)
from flame.task_registry import supported

pytestmark = pytest.mark.prompts


def test_prompt_registry_basic_functionality():
    """Test basic prompt registry operations."""
    # Test getting a valid prompt
    prompt_fn = get_prompt("fomc", PromptFormat.ZERO_SHOT)
    assert prompt_fn is not None
    assert callable(prompt_fn)

    # Test the prompt function works
    result = prompt_fn("Test input text")
    assert isinstance(result, str)
    assert "Test input text" in result

    # Test that few-shot might be different or same
    few_shot_fn = get_prompt("fomc", PromptFormat.FEW_SHOT)
    assert few_shot_fn is not None
    assert callable(few_shot_fn)


def test_prompt_availability():
    """Test that prompts are available for common tasks."""
    # Check some expected tasks
    expected_tasks = ["fomc", "finqa", "headlines", "finer", "fpb"]

    for task in expected_tasks:
        prompt_fn = get_prompt(task, PromptFormat.ZERO_SHOT)
        assert prompt_fn is not None, f"No prompt found for task: {task}"
        assert callable(prompt_fn)


def test_extraction_prompts():
    """Test extraction prompt functionality."""
    # Test extraction format prompts
    extraction_tasks = ["numclaim", "fnxl", "subjectiveqa"]

    for task in extraction_tasks:
        prompt_fn = get_prompt(task, PromptFormat.EXTRACTION)
        if prompt_fn is not None:  # Some tasks might not have extraction prompts
            assert callable(prompt_fn)
            # Test with appropriate arguments based on the task
            if task == "subjectiveqa":
                result = prompt_fn("test response", "feature1")
            else:
                result = prompt_fn("test response")
            assert isinstance(result, str)
            assert len(result) > 10


def test_prompt_task_alignment():
    """Ensure prompt registry aligns with task registry."""
    # Get all supported inference tasks
    inference_tasks = supported("inference")

    # Count tasks that have prompts
    tasks_with_prompts = 0
    for task in inference_tasks:
        prompt_fn = get_prompt(task, PromptFormat.ZERO_SHOT)
        if prompt_fn is not None:
            tasks_with_prompts += 1

    # Check that most inference tasks have prompts
    assert tasks_with_prompts >= 10, f"Only {tasks_with_prompts} tasks have prompts"

    # Test that we can use prompts for a sample of tasks
    sample_tasks = ["fomc", "finqa", "headlines", "finer", "fpb"]
    for task in sample_tasks:
        if task in inference_tasks:
            prompt_fn = get_prompt(task, PromptFormat.ZERO_SHOT)
            assert prompt_fn is not None
            assert callable(prompt_fn)


def test_prompt_format_enum():
    """Test the PromptFormat enum."""
    # Test that enum values exist
    assert hasattr(PromptFormat, "DEFAULT")
    assert hasattr(PromptFormat, "ZERO_SHOT")
    assert hasattr(PromptFormat, "FEW_SHOT")
    assert hasattr(PromptFormat, "EXTRACTION")

    # Test that they are different values
    assert PromptFormat.ZERO_SHOT != PromptFormat.FEW_SHOT
    assert PromptFormat.DEFAULT != PromptFormat.EXTRACTION


@pytest.mark.parametrize("task", ["fomc", "finqa", "headlines", "finer", "fpb"])
def test_common_prompts(task):
    """Test commonly used prompts work correctly."""
    prompt_fn = get_prompt(task, PromptFormat.ZERO_SHOT)
    assert prompt_fn is not None

    # Test with sample input
    result = prompt_fn("Sample input for testing")
    assert isinstance(result, str)
    assert len(result) > 50  # Should generate substantial prompt

    # Ensure no template variables left
    assert "{{" not in result
    assert "}}" not in result


def test_invalid_prompt_requests():
    """Test handling of invalid prompt requests."""
    # Non-existent task
    assert get_prompt("fake_task", PromptFormat.ZERO_SHOT) is None

    # Test get_prompt_by_name with invalid name
    assert get_prompt_by_name("fake_prompt_name") is None
