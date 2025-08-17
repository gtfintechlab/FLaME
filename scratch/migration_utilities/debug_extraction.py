#!/usr/bin/env python3
"""Debug extraction issues with FOMC responses."""

import re
import sys

sys.path.insert(0, "/home/gmatlin/Codespace/FLAME/benchforge")
from bench_forge.prompts.extractor import ResponseExtractor, ExtractionStrategy

# Actual model responses from the test
test_responses = [
    # Response 1 - from zero_shot
    """The correct classification is:

NEUTRAL

The""",
    # Response 2 - typical response
    """Classification: HAWKISH""",
    # Response 3 - Another format
    """DOVISH""",
    # Response 4 - Chain of thought
    """To analyze the given Federal Open Market Committee (FOMC) statement and classify it as HAWKISH, DOVISH, or NEUTRAL, let's follow the steps:

1. **Identify key monetary policy indicators (rates, inflation,""",
    # Response 5 - More complete
    """The statement indicates a HAWKISH stance because it mentions raising rates.""",
]


def test_extraction():
    """Test extraction with different strategies."""
    extractor = ResponseExtractor()
    valid_labels = ["HAWKISH", "DOVISH", "NEUTRAL"]

    print("=" * 60)
    print("Testing Extraction Strategies")
    print("=" * 60)

    for i, response in enumerate(test_responses, 1):
        print(f"\nTest {i}:")
        print(f"Response: {response[:100]}...")

        # Try KEYWORD strategy (default)
        result = extractor.extract_label(
            response, valid_labels, strategy=ExtractionStrategy.KEYWORD
        )
        print(f"  KEYWORD: {result}")

        # Try REGEX strategy
        result = extractor.extract_label(
            response, valid_labels, strategy=ExtractionStrategy.REGEX
        )
        print(f"  REGEX: {result}")

        # Try FUZZY strategy
        result = extractor.extract_label(
            response, valid_labels, strategy=ExtractionStrategy.FUZZY
        )
        print(f"  FUZZY: {result}")

        # Try without specifying strategy (will try all)
        result = extractor.extract_label(response, valid_labels)
        print(f"  AUTO: {result}")

        # Manual check
        response_upper = response.upper()
        manual_result = None
        for label in valid_labels:
            if label in response_upper:
                manual_result = label
                break
        print(f"  MANUAL: {manual_result}")


def test_case_sensitivity():
    """Test case sensitivity issues."""
    print("\n" + "=" * 60)
    print("Testing Case Sensitivity")
    print("=" * 60)

    response = "The correct classification is:\n\nNEUTRAL\n\nThe"
    valid_labels = ["HAWKISH", "DOVISH", "NEUTRAL"]

    # Case-insensitive extractor (default)
    extractor_insensitive = ResponseExtractor(case_sensitive=False)
    result = extractor_insensitive.extract_label(response, valid_labels)
    print(f"Case-insensitive: {result}")

    # Case-sensitive extractor
    extractor_sensitive = ResponseExtractor(case_sensitive=True)
    result = extractor_sensitive.extract_label(response, valid_labels)
    print(f"Case-sensitive: {result}")

    # Check the actual matching logic
    text_lower = response.lower()
    for label in valid_labels:
        label_lower = label.lower()
        pattern = r"\b" + re.escape(label_lower) + r"\b"
        match = re.search(pattern, text_lower)
        if match:
            print(
                f"Found '{label}' with pattern '{pattern}' at position {match.span()}"
            )


def test_model_response_formats():
    """Test different response formats the model might produce."""
    print("\n" + "=" * 60)
    print("Testing Model Response Formats")
    print("=" * 60)

    extractor = ResponseExtractor()
    valid_labels = ["HAWKISH", "DOVISH", "NEUTRAL"]

    # Different formats the model might use
    formats = [
        "HAWKISH",
        "hawkish",
        "Hawkish",
        "Classification: HAWKISH",
        "The answer is HAWKISH",
        "Answer: HAWKISH",
        "HAWKISH\n",
        "\nHAWKISH\n",
        "The correct classification is:\n\nHAWKISH\n\nThe",
        "Based on the analysis, this is HAWKISH",
        "(HAWKISH)",
        '"HAWKISH"',
        "'HAWKISH'",
    ]

    for format_str in formats:
        result = extractor.extract_label(format_str, valid_labels)
        status = "✓" if result else "✗"
        print(f"{status} '{format_str[:30]}...' -> {result}")


if __name__ == "__main__":
    test_extraction()
    test_case_sensitivity()
    test_model_response_formats()
