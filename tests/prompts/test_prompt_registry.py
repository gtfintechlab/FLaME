"""Test suite for the unified prompt registry.

This test suite ensures that our prompt registry:
1. Contains all prompt functions
2. Returns the correct prompt functions for each task and format
3. Correctly handles various edge cases
"""

import pytest

# Import our registry
from flame.code.prompts import get_prompt, PromptFormat
from flame.code.prompts.registry import list_tasks, _REGISTRY

pytestmark = pytest.mark.prompts


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
    ]

    for task, format_type, test_input in test_cases:
        func = get_prompt(task, format_type)
        assert func is not None, f"Function for {task}/{format_type} not found"
        result = func(test_input)
        assert isinstance(result, str), f"Result for {task}/{format_type} not a string"
        assert (
            test_input in result
        ), f"Input not found in result for {task}/{format_type}"

    # Test few-shot stubs separately
    few_shot_stubs = [
        ("banking77", PromptFormat.FEW_SHOT, "test input"),
        ("numclaim", PromptFormat.FEW_SHOT, "test input"),
    ]

    for task, format_type, test_input in few_shot_stubs:
        func = get_prompt(task, format_type)
        assert func is not None, f"Function for {task}/{format_type} not found"
        result = func(test_input)
        assert result is None, f"Few-shot stub for {task} should return None"


def test_extraction_prompt_registry():
    """Test that extraction prompts are properly registered and functional."""
    # Test cases: (task, expected_function_name)
    extraction_tasks = [
        ("fpb", "fpb_extraction_prompt"),
        ("headlines", "headlines_extraction_prompt"),
        ("refind", "refind_extraction_prompt"),
        ("fomc", "fomc_extraction_prompt"),
        ("numclaim", "numclaim_extraction_prompt"),
        ("finentity", "finentity_extraction_prompt"),
        ("finer", "finer_extraction_prompt"),
        ("causal_classification", "causal_classifciation_extraction_prompt"),
        ("causal_detection", "causal_detection_extraction_prompt"),
        ("banking77", "banking_77_extraction_prompt"),
        ("finbench", "finbench_extraction_prompt"),
        ("qa", "qa_extraction_prompt"),
        ("finred", "finred_extraction_prompt"),
        ("fiqa_task1", "fiqa_1_extraction_prompt"),
        ("subjectiveqa", "subjectiveqa_extraction_prompt"),
        ("fnxl", "fnxl_extraction_prompt"),
        ("convfinqa", "convfinqa_extraction_prompt"),
    ]

    for task, expected_name in extraction_tasks:
        func = get_prompt(task, PromptFormat.EXTRACTION)
        assert func is not None, f"No extraction prompt found for {task}"
        assert func.__name__ == expected_name, (
            f"Wrong function name for {task} extraction prompt. "
            f"Expected {expected_name}, got {func.__name__}"
        )


def test_task_registry_extraction_prompt_alignment():
    """Test that all evaluation tasks in the registry have extraction prompts if needed."""
    from flame.task_registry import EVALUATE_MAP

    # Tasks known to not use extraction prompts (they use other evaluation methods)
    tasks_without_extraction = {
        "bizbench",  # Uses different evaluation approach
        "ectsum",  # Summarization task, uses ROUGE metrics
        "edtsum",  # Summarization task, uses ROUGE metrics
        "fiqa_task2",  # Uses different evaluation approach
        "mmlu",  # Multiple choice, extracts letter directly
        "tatqa",  # Uses different evaluation approach
    }

    # Tasks that use the generic 'qa' extraction prompt
    tasks_using_qa_prompt = {"finqa"}

    missing_prompts = []

    for task_name in EVALUATE_MAP:
        if task_name in tasks_without_extraction:
            continue

        # Check for task-specific extraction prompt
        prompt_task = "qa" if task_name in tasks_using_qa_prompt else task_name
        func = get_prompt(prompt_task, PromptFormat.EXTRACTION)

        if func is None:
            missing_prompts.append(task_name)

    assert (
        len(missing_prompts) == 0
    ), f"The following evaluation tasks are missing extraction prompts: {missing_prompts}"


def test_extraction_prompt_functionality():
    """Test that extraction prompts work correctly with sample inputs."""
    test_cases = [
        # (task, sample_llm_response, expected_contains, extra_args)
        (
            "fpb",
            "The sentiment is POSITIVE because...",
            ["POSITIVE", "NEGATIVE", "NEUTRAL"],
            {},
        ),
        (
            "headlines",
            "negative sentiment",
            ["Price_or_Not", "Direction_Up", "JSON object"],
            {},
        ),
        ("fomc", "hawkish tone", ["HAWKISH", "DOVISH"], {}),
        ("refind", "B-LOCATION", ["PERSON-TITLE", "ORG-ORG", "NO-REL"], {}),
        ("numclaim", "0 or 1", ["INCLAIM", "OUTOFCLAIM"], {}),
        (
            "banking77",
            "I need to activate my card",
            ["activate_my_card", "banking intents"],
            {},
        ),
        (
            "causal_classification",
            "The label is 1",
            ["0, 1, or 2", "single number"],
            {},
        ),
        (
            "causal_detection",
            "B-CAUSE I-CAUSE O",
            ["O", "I-CAUSE", "B-CAUSE", "I-EFFECT", "B-EFFECT"],
            {},
        ),
        (
            "convfinqa",
            "The answer is 42.5",
            ["numerical value", "integer, decimal, or percentage"],
            {},
        ),
        ("finbench", "This is HIGH RISK", ["HIGH RISK", "LOW RISK"], {}),
        (
            "finentity",
            "Apple Inc. is a company",
            ["entity", "JSON", "value", "tag", "label"],
            {},
        ),
        ("finer", "John works at Apple", ["O", "PER_B", "ORG_B", "numeric values"], {}),
        (
            "finred",
            "Company X owns Company Y",
            ["subsidiary", "owned_by", "NO-REL"],
            {},
        ),
        ("fiqa_task1", "Sentiment score: 0.5", ["sentiment score", "-1 and 1"], {}),
        (
            "fnxl",
            "Revenue: $1000",
            ["list_of_numerical_values", "xbrl_tag", "JSON"],
            {},
        ),
        (
            "qa",
            "The final answer is 42",
            ["final answer", "integer, decimal, percentage"],
            {},
        ),
        (
            "subjectiveqa",
            "Rating is 2",
            ["0, 1, or 2", "rating"],
            {"feature": "CLARITY"},
        ),
    ]

    for test_case in test_cases:
        task = test_case[0]
        sample_response = test_case[1]
        expected_contains = test_case[2]
        extra_args = test_case[3] if len(test_case) > 3 else {}

        # Special case: finqa uses 'qa' extraction prompt
        prompt_task = "qa" if task == "finqa" else task

        func = get_prompt(prompt_task, PromptFormat.EXTRACTION)
        assert func is not None, f"No extraction prompt found for {task}"

        # Call with extra args if needed (e.g., subjectiveqa needs feature parameter)
        result = func(sample_response, **extra_args)
        assert isinstance(
            result, str
        ), f"Extraction prompt for {task} should return a string"
        assert (
            len(result) > 0
        ), f"Extraction prompt for {task} should not return empty string"

        # Check that the prompt contains expected elements
        for expected in expected_contains:
            assert (
                expected in result
            ), f"Extraction prompt for {task} should contain '{expected}'"


def test_extraction_prompt_integration():
    """Test that extraction prompts integrate correctly with evaluation modules."""
    # Import a few evaluation modules to ensure they can access extraction prompts
    from flame.code.fpb.fpb_evaluate import fpb_evaluate
    from flame.code.banking77.banking77_evaluate import banking77_evaluate
    from flame.code.finred.finred_evaluate import finred_evaluate

    # These imports should not fail, and the modules should have access to the registry
    assert fpb_evaluate is not None
    assert banking77_evaluate is not None
    assert finred_evaluate is not None

    # Verify that constants are properly imported
    from flame.code.prompts.constants import (
        banking77_list,
        finred_extraction_labels,
        refind_possible_relationships,
    )

    assert len(banking77_list) == 77, "banking77_list should have 77 categories"
    assert (
        len(finred_extraction_labels) == 29
    ), "finred_extraction_labels should have 29 labels"
    assert (
        len(refind_possible_relationships) == 8
    ), "refind_possible_relationships should have 8 relationships"
