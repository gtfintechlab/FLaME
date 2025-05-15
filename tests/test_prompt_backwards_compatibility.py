"""
Test suite for prompt backward compatibility.

This test suite verifies that:
1. The old imports from the original prompt files still work
2. Functions from the new and old paths are identical
3. The functions behave correctly
"""


# Import functions from both old and new paths
def test_zeroshot_backwards_compatibility():
    """Test that zero-shot prompt imports from legacy module still work."""
    # Import from old path
    from flame.code.prompts_zeroshot import (
        fpb_zeroshot_prompt,
        headlines_zeroshot_prompt,
        numclaim_zeroshot_prompt,
        fomc_zeroshot_prompt,
        banking77_zeroshot_prompt,
        numclaim_prompt,  # Test aliased function
    )

    # Import from new path for comparison
    from flame.code.prompts import (
        fpb_zeroshot_prompt as new_fpb_prompt,
        headlines_zeroshot_prompt as new_headlines_prompt,
        numclaim_zeroshot_prompt as new_numclaim_prompt,
        fomc_zeroshot_prompt as new_fomc_prompt,
        banking77_zeroshot_prompt as new_banking77_prompt,
    )

    # Check that functions are the same objects
    assert fpb_zeroshot_prompt is new_fpb_prompt
    assert headlines_zeroshot_prompt is new_headlines_prompt
    assert numclaim_zeroshot_prompt is new_numclaim_prompt
    assert fomc_zeroshot_prompt is new_fomc_prompt
    assert banking77_zeroshot_prompt is new_banking77_prompt

    # Check that aliased functions work
    assert numclaim_prompt is numclaim_zeroshot_prompt

    # Test function behavior
    test_input = "This is a test sentence."
    assert isinstance(fpb_zeroshot_prompt(test_input), str)
    assert test_input in fpb_zeroshot_prompt(test_input)


def test_fewshot_backwards_compatibility():
    """Test that few-shot prompt imports from legacy module still work."""
    # Import from old path
    from flame.code.prompts_fewshot import (
        banking77_fewshot_prompt,
        numclaim_fewshot_prompt,
        fpb_fewshot_prompt,
        fomc_fewshot_prompt,
    )

    # Import from new path for comparison
    from flame.code.prompts import (
        banking77_fewshot_prompt as new_banking77_prompt,
        numclaim_fewshot_prompt as new_numclaim_prompt,
        fpb_fewshot_prompt as new_fpb_prompt,
        fomc_fewshot_prompt as new_fomc_prompt,
    )

    # Check that functions are the same objects
    assert banking77_fewshot_prompt is new_banking77_prompt
    assert numclaim_fewshot_prompt is new_numclaim_prompt
    assert fpb_fewshot_prompt is new_fpb_prompt
    assert fomc_fewshot_prompt is new_fomc_prompt

    # Test banking77 function behavior (it's the only one fully implemented)
    test_input = "I need to change my PIN"
    result = banking77_fewshot_prompt(test_input)
    assert isinstance(result, str)
    assert test_input in result

    # Test stub behavior
    assert numclaim_fewshot_prompt("test") is None
