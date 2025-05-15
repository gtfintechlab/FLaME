"""
Test suite for the new prompts package structure.

This test suite verifies that:
1. The registry correctly maps task and format to functions
2. Direct imports work as expected
3. Function registration is working correctly
"""

from flame.code.prompts import (
    get_prompt,
    PromptFormat,
    bizbench_prompt,
    banking77_zeroshot_prompt,
    banking77_fewshot_prompt,
    fpb_zeroshot_prompt,
)


def test_registry_task_lookup():
    """Test that get_prompt returns the correct function for each task and format."""
    # Test zero-shot lookups
    assert get_prompt("headlines", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("numclaim", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("fomc", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("fpb", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("banking77", PromptFormat.ZERO_SHOT) is not None

    # Test few-shot lookups
    assert get_prompt("banking77", PromptFormat.FEW_SHOT) is not None

    # Test default lookups
    assert get_prompt("bizbench", PromptFormat.DEFAULT) is not None
    assert get_prompt("econlogicqa", PromptFormat.DEFAULT) is not None


def test_direct_imports():
    """Test that functions can be imported directly."""
    # Direct imports should be functions
    assert callable(bizbench_prompt)
    assert callable(banking77_zeroshot_prompt)
    assert callable(banking77_fewshot_prompt)
    assert callable(fpb_zeroshot_prompt)


def test_function_behavior():
    """Test that functions behave correctly."""
    # Test a few key functions
    test_input = "This is a test sentence."

    # Zero-shot
    zero_shot_result = banking77_zeroshot_prompt(test_input)
    assert isinstance(zero_shot_result, str)
    assert test_input in zero_shot_result

    # Few-shot
    few_shot_result = banking77_fewshot_prompt(test_input)
    assert isinstance(few_shot_result, str)
    assert test_input in few_shot_result

    # Base prompt
    base_result = bizbench_prompt(test_input)
    assert isinstance(base_result, str)
    assert test_input in base_result


def test_registry_function_equivalence():
    """Test that registry functions are the same as direct imports."""
    # Functions from registry should be identical to direct imports
    assert get_prompt("bizbench", PromptFormat.DEFAULT) is bizbench_prompt
    assert get_prompt("banking77", PromptFormat.ZERO_SHOT) is banking77_zeroshot_prompt
    assert get_prompt("banking77", PromptFormat.FEW_SHOT) is banking77_fewshot_prompt
    assert get_prompt("fpb", PromptFormat.ZERO_SHOT) is fpb_zeroshot_prompt
