"""
Test suite for prompt aliases.

This test suite verifies that prompt aliases are working correctly.

BACKGROUND:
-----------
The FLaME codebase has transitioned to using a registry-based prompt system
where prompts are accessed via get_prompt(task_name, PromptFormat).
However, for backward compatibility, aliases were created that map shorter
names (e.g., 'numclaim_prompt') to their full names (e.g., 'numclaim_zeroshot_prompt').

These aliases exist in src/flame/code/prompts/__init__.py (lines 148-166)
and allow code to use the shorter names for common zero-shot prompts.

STATUS:
-------
- The main codebase has been updated to use the registry system
- All inference files now use get_prompt() instead of direct imports
- These aliases remain for potential backward compatibility with:
  - External scripts that might import these names
  - Jupyter notebooks or other code not in the repository
  - Any legacy code that hasn't been updated yet

TODO:
-----
Once we confirm that no external dependencies use these aliases:
1. Remove the aliases from src/flame/code/prompts/__init__.py
2. Delete this test file
3. Update any remaining code that uses the short names

For now, these tests ensure the aliases continue to work correctly.
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

    # Test stub behavior - all few-shot prompts are stubs for now
    assert banking77_fewshot_prompt("test") is None
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
