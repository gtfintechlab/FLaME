"""Test suite for the unified prompt registry.

This test suite ensures that our prompt registry:
1. Contains all prompt functions
2. Returns the correct prompt functions for each task and format
3. Correctly handles various edge cases
"""

# Import our registry
from flame.code.prompts import get_prompt, PromptFormat
from flame.code.prompts.registry import list_tasks, _REGISTRY


def test_get_prompt_returns_correct_function():
    """Test that get_prompt returns the correct function for each task and format."""
    # Test cases: (task, format, expected_name)
    test_cases = [
        # Zero-shot cases
        ("fpb", PromptFormat.ZERO_SHOT, "fpb_zeroshot_prompt"),
        ("numclaim", PromptFormat.ZERO_SHOT, "numclaim_zeroshot_prompt"),
        ("banking77", PromptFormat.ZERO_SHOT, "banking77_zeroshot_prompt"),
        # Few-shot cases
        ("banking77", PromptFormat.FEW_SHOT, "banking77_fewshot_prompt"),
        # Default cases (using aliases)
        ("fpb", PromptFormat.DEFAULT, "fpb_zeroshot_prompt"),
        ("numclaim", PromptFormat.DEFAULT, "numclaim_zeroshot_prompt"),
    ]

    for task, format_type, expected_name in test_cases:
        func = get_prompt(task, format_type)
        assert func is not None, f"No prompt function found for {task}, {format_type}"
        assert (
            func.__name__ == expected_name
        ), f"Wrong function name for {task}, {format_type}"


def test_fallback_behavior():
    """Test that get_prompt falls back to available formats when requested format is not found."""
    # Find a task with both DEFAULT and ZERO_SHOT registered
    # First, check if there's a task with only ZERO_SHOT and no DEFAULT
    zero_shot_only_task = None
    for task in _REGISTRY:
        if (
            PromptFormat.ZERO_SHOT in _REGISTRY[task]
            and PromptFormat.DEFAULT not in _REGISTRY[task]
        ):
            zero_shot_only_task = task
            break

    # If we found a task with only ZERO_SHOT, test fallback
    if zero_shot_only_task:
        func = get_prompt(zero_shot_only_task, PromptFormat.DEFAULT)
        assert func is not None
        assert func.__name__.endswith(
            "_zeroshot_prompt"
        ), f"Function name should end with _zeroshot_prompt, got {func.__name__}"

    # If a specific format is requested and not available, should return None
    # Find a task with no FEW_SHOT format
    for task in _REGISTRY:
        if PromptFormat.FEW_SHOT not in _REGISTRY[task]:
            assert (
                get_prompt(task, PromptFormat.FEW_SHOT) is None
            ), f"Expected None for {task} with FEW_SHOT format"
            break


def test_list_tasks():
    """Test that list_tasks returns all registered tasks and formats."""
    tasks = list_tasks()

    # Check a few known tasks are included
    assert "fpb" in tasks
    assert "numclaim" in tasks
    assert "banking77" in tasks

    # Check that formats are correctly identified
    # Note: The format names in the output may vary based on implementation
    # Just ensure that the task is registered with at least one format
    assert len(tasks["fpb"]) > 0
    assert len(tasks["banking77"]) > 0


def test_prompt_function_behavior():
    """Test that prompt functions from the registry behave the same as originals."""
    # Test some representative prompt functions
    test_cases = [
        # (task, format, input, expected_contains, extra_args)
        ("numclaim", PromptFormat.ZERO_SHOT, "Sample text", "Sample text", {}),
        (
            "banking77",
            PromptFormat.ZERO_SHOT,
            "I need to change my pin",
            "change my pin",
            {},
        ),
    ]

    for task, format_type, input_text, expected_contains, extra_args in test_cases:
        func = get_prompt(task, format_type)
        assert func is not None

        # Call the function and check the result
        result = func(input_text, **extra_args)
        assert isinstance(result, str)
        assert expected_contains in result

    # Special case for fpb which requires prompt_format
    fpb_func = get_prompt("fpb", PromptFormat.ZERO_SHOT)
    assert fpb_func is not None
    result = fpb_func("Sample text", prompt_format="flame")
    assert isinstance(result, str)
    assert "Sample text" in result


def test_registry_handles_task_format_combinations():
    """Test that the registry correctly handles a few common task/format combinations."""
    # Test specific known-good combinations
    test_cases = [
        ("fpb", PromptFormat.ZERO_SHOT, "test input"),
        ("headlines", PromptFormat.ZERO_SHOT, "test input"),
        ("numclaim", PromptFormat.ZERO_SHOT, "test input"),
        ("banking77", PromptFormat.ZERO_SHOT, "test input"),
        ("banking77", PromptFormat.FEW_SHOT, "test input"),
    ]

    for task, format_type, test_input in test_cases:
        func = get_prompt(task, format_type)
        assert func is not None, f"Function for {task}/{format_type} not found"
        result = func(test_input)
        assert isinstance(result, str), f"Result for {task}/{format_type} not a string"
        assert (
            test_input in result
        ), f"Input not found in result for {task}/{format_type}"
