"""
Test suite for the new prompts package structure.

This test suite verifies that:
1. The registry correctly maps task and format to functions
2. Direct imports work as expected
3. Function registration is working correctly
"""

import pytest

from flame.code.prompts import (
    get_prompt,
    PromptFormat,
    bizbench_prompt,
    banking77_zeroshot_prompt,
    banking77_fewshot_prompt,
    fpb_zeroshot_prompt,
    fiqa_task1_zeroshot_prompt,
    fiqa_task2_zeroshot_prompt,
    finer_zeroshot_prompt,
    finentity_zeroshot_prompt,
    finbench_zeroshot_prompt,
)

pytestmark = pytest.mark.prompts


def test_registry_task_lookup():
    """Test that get_prompt returns the correct function for each task and format."""
    # Test zero-shot lookups
    assert get_prompt("headlines", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("numclaim", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("fomc", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("fpb", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("banking77", PromptFormat.ZERO_SHOT) is not None

    # Test previously migrated prompts
    assert get_prompt("fiqa_task1", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("fiqa_task2", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("finer", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("finentity", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("finbench", PromptFormat.ZERO_SHOT) is not None

    # Test newly migrated prompts
    assert get_prompt("ectsum", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("finqa", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("convfinqa", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("tatqa", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("causal_classification", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("finred", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("causal_detection", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("subjectiveqa", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("fnxl", PromptFormat.ZERO_SHOT) is not None
    assert get_prompt("refind", PromptFormat.ZERO_SHOT) is not None

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

    # Newly migrated prompts should be callable
    assert callable(fiqa_task1_zeroshot_prompt)
    assert callable(fiqa_task2_zeroshot_prompt)
    assert callable(finer_zeroshot_prompt)
    assert callable(finentity_zeroshot_prompt)
    assert callable(finbench_zeroshot_prompt)


def test_function_behavior():
    """Test that functions behave correctly."""
    # Test a few key functions
    test_input = "This is a test sentence."

    # Zero-shot
    zero_shot_result = banking77_zeroshot_prompt(test_input)
    assert isinstance(zero_shot_result, str)
    assert test_input in zero_shot_result

    # Few-shot (currently a stub)
    few_shot_result = banking77_fewshot_prompt(test_input)
    assert few_shot_result is None  # Stub returns None

    # Base prompt
    base_result = bizbench_prompt(test_input)
    assert isinstance(base_result, str)
    assert test_input in base_result

    # Newly migrated prompts
    fiqa_task1_result = fiqa_task1_zeroshot_prompt(test_input)
    assert isinstance(fiqa_task1_result, str)
    assert "sentiment" in fiqa_task1_result
    assert test_input in fiqa_task1_result

    finer_result = finer_zeroshot_prompt(test_input)
    assert isinstance(finer_result, str)
    assert "named entity" in finer_result
    assert test_input in finer_result


def test_registry_function_equivalence():
    """Test that registry functions are the same as direct imports."""
    # Functions from registry should be identical to direct imports
    assert get_prompt("bizbench", PromptFormat.DEFAULT) is bizbench_prompt
    assert get_prompt("banking77", PromptFormat.ZERO_SHOT) is banking77_zeroshot_prompt
    assert get_prompt("banking77", PromptFormat.FEW_SHOT) is banking77_fewshot_prompt
    assert get_prompt("fpb", PromptFormat.ZERO_SHOT) is fpb_zeroshot_prompt

    # Newly migrated prompts should be properly registered
    assert (
        get_prompt("fiqa_task1", PromptFormat.ZERO_SHOT) is fiqa_task1_zeroshot_prompt
    )
    assert (
        get_prompt("fiqa_task2", PromptFormat.ZERO_SHOT) is fiqa_task2_zeroshot_prompt
    )
    assert get_prompt("finer", PromptFormat.ZERO_SHOT) is finer_zeroshot_prompt
    assert get_prompt("finentity", PromptFormat.ZERO_SHOT) is finentity_zeroshot_prompt
    assert get_prompt("finbench", PromptFormat.ZERO_SHOT) is finbench_zeroshot_prompt
