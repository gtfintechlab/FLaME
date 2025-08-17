#!/usr/bin/env python3
"""Test BenchForge extraction safely - loads keys from .env file."""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify keys are loaded (without printing them!)
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    print("ERROR: HUGGINGFACEHUB_API_TOKEN not found in environment")
    sys.exit(1)
if not os.getenv("TOGETHERAI_API_KEY"):
    print("ERROR: TOGETHERAI_API_KEY not found in environment")
    sys.exit(1)

print("✓ API keys loaded from .env file")

# Setup paths
sys.path.insert(0, "/home/gmatlin/Codespace/FLAME")
sys.path.insert(0, "/home/gmatlin/Codespace/FLAME/benchforge")


def test_extraction_logic():
    """Test the extraction logic only."""
    from bench_forge.flame.tasks.fomc import FOMCTask, FOMCConfig

    config = FOMCConfig(name="fomc")
    task = FOMCTask(config)

    test_cases = [
        ("The correct classification is:\n\nNEUTRAL", "NEUTRAL"),
        ("HAWKISH", "HAWKISH"),
        ("I would classify the statement as DOVISH.", "DOVISH"),
        ("Based on the analysis, this is NEUTRAL", "NEUTRAL"),
        ("Classification: HAWKISH\n\nThe statement suggests...", "HAWKISH"),
    ]

    print("\nTesting extraction logic:")
    print("=" * 60)

    success_count = 0
    for response, expected in test_cases:
        extracted = task.extract_label_from_response(response)
        status = "✓" if extracted == expected else "✗"
        print(f"{status} '{response[:40]}...' -> {extracted} (expected: {expected})")
        if extracted == expected:
            success_count += 1

    print("=" * 60)
    print(f"Extraction test: {success_count}/{len(test_cases)} passed")

    return success_count == len(test_cases)


if __name__ == "__main__":
    success = test_extraction_logic()
    sys.exit(0 if success else 1)
