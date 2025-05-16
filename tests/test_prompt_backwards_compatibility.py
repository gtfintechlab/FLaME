"""
Test suite for prompt functionality.

This test suite verifies that:
1. Prompt functions work correctly
2. Functions behave as expected
3. All required prompts exist in the new package
"""


def test_zeroshot_prompt_functionality():
    """Test that zero-shot prompt functions work correctly."""
    # Import from new path
    from flame.code.prompts import (
        # Zero-shot prompt functions
        fpb_zeroshot_prompt,
        numclaim_zeroshot_prompt,
        ectsum_zeroshot_prompt,
        finqa_zeroshot_prompt,
        causal_detection_zeroshot_prompt,
        numclaim_prompt,
        ectsum_prompt,
        finqa_prompt,
    )

    # Check that aliased functions work
    assert numclaim_prompt is numclaim_zeroshot_prompt
    assert ectsum_prompt is ectsum_zeroshot_prompt
    assert finqa_prompt is finqa_zeroshot_prompt

    # Test function behavior
    test_input = "This is a test sentence."
    assert isinstance(fpb_zeroshot_prompt(test_input), str)
    assert test_input in fpb_zeroshot_prompt(test_input)

    # Test a few representative functions
    document = "This is a test document for summarization."
    tokens = ["This", "is", "a", "test"]

    assert isinstance(ectsum_zeroshot_prompt(document), str)
    assert document in ectsum_zeroshot_prompt(document)

    assert isinstance(finqa_zeroshot_prompt(document), str)
    assert document in finqa_zeroshot_prompt(document)

    assert isinstance(causal_detection_zeroshot_prompt(tokens), str)
    assert "token" in causal_detection_zeroshot_prompt(tokens).lower()


def test_fewshot_prompt_functionality():
    """Test that few-shot prompt functions work correctly."""
    # Import directly from the new path
    from flame.code.prompts import (
        banking77_fewshot_prompt,
        numclaim_fewshot_prompt,
    )

    # Test banking77 function behavior (it's the only one fully implemented)
    test_input = "I need to change my PIN"
    result = banking77_fewshot_prompt(test_input)
    assert isinstance(result, str)
    assert test_input in result

    # Test stub behavior
    assert numclaim_fewshot_prompt("test") is None


def test_registry_access():
    """Test that prompts can be accessed through the registry."""
    from flame.code.prompts import get_prompt, PromptFormat

    # Test retrieving various prompt functions
    fpb_fn = get_prompt("fpb", PromptFormat.ZERO_SHOT)
    assert fpb_fn is not None
    assert fpb_fn.__name__ == "fpb_zeroshot_prompt"

    # Test function output
    test_input = "This is a test."
    output = fpb_fn(test_input)
    assert isinstance(output, str)
    assert test_input in output
