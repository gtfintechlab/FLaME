"""
Consolidated test suite for prompt functionality.

Tests the prompt registry system that enables task-specific prompt generation
for the FLaME framework.
"""

import pytest

from flame.code.prompts import (
    PromptFormat,
    get_extraction_prompt,
    get_prompt,
    list_tasks,
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

    # Test fallback behavior
    few_shot_fn = get_prompt("fomc", PromptFormat.FEW_SHOT)
    assert few_shot_fn is not None
    # Should fall back to zero-shot
    assert few_shot_fn == prompt_fn


def test_list_tasks():
    """Test listing available tasks."""
    available_tasks = list_tasks()
    assert isinstance(available_tasks, set)
    assert len(available_tasks) > 0

    # Check some expected tasks
    expected = {"fomc", "finqa", "headline", "ner", "fpb"}
    assert expected.issubset(available_tasks)


def test_extraction_prompts():
    """Test extraction prompt functionality."""
    # Test get_extraction_prompt
    prompt_fn = get_extraction_prompt("extraction_prompt_detailed")
    assert prompt_fn is not None
    assert callable(prompt_fn)

    # Test the prompt works
    result = prompt_fn("Extract key info", "Document text here")
    assert isinstance(result, str)
    assert "Extract key info" in result
    assert "Document text here" in result


def test_prompt_task_alignment():
    """Ensure prompt registry aligns with task registry."""
    # Get all supported inference tasks
    inference_tasks = supported("inference")
    prompt_tasks = list_tasks()

    # Check that most inference tasks have prompts
    # (Some tasks might not need prompts)
    tasks_with_prompts = inference_tasks.intersection(prompt_tasks)
    assert len(tasks_with_prompts) >= 10

    # Test that we can get prompts for these tasks
    for task in list(tasks_with_prompts)[:5]:  # Test a sample
        prompt_fn = get_prompt(task, PromptFormat.ZERO_SHOT)
        assert prompt_fn is not None
        assert callable(prompt_fn)


def test_prompt_format_enum():
    """Test the PromptFormat enum."""
    assert PromptFormat.ZERO_SHOT.value == "zero_shot"
    assert PromptFormat.FEW_SHOT.value == "few_shot"
    assert PromptFormat.CHAIN_OF_THOUGHT.value == "chain_of_thought"

    # Test string conversion
    assert str(PromptFormat.ZERO_SHOT) == "PromptFormat.ZERO_SHOT"


@pytest.mark.parametrize("task", ["fomc", "finqa", "headline", "ner", "fpb"])
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

    # Invalid extraction prompt
    assert get_extraction_prompt("fake_extraction") is None
