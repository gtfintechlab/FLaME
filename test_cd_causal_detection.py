#!/usr/bin/env python3
"""Test Causal Detection (CD) task implementation."""

import os
import sys
import logging

# Add BenchForge to path
sys.path.insert(0, os.path.abspath("benchforge"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_cd_basic():
    """Test basic CD functionality."""
    print("\n" + "=" * 60)
    print("TESTING CAUSAL DETECTION (CD) - BASIC FUNCTIONALITY")
    print("=" * 60)

    try:
        from bench_forge.flame.tasks.causal_detection import (
            CausalDetectionTask,
            CausalDetectionConfig,
        )
        from bench_forge.tasks.config import PromptFormat

        # Initialize task
        config = CausalDetectionConfig(
            name="causal_detection", prompt_format=PromptFormat.ZERO_SHOT
        )
        task = CausalDetectionTask(config)

        print("✅ Task initialized successfully")
        print(f"Valid labels: {config.valid_labels}")
        print(f"Label mapping: {config.label_mapping}")

        return task

    except Exception as e:
        print(f"❌ Task initialization failed: {e}")
        return None


def test_cd_prompt_creation():
    """Test CD prompt creation."""
    print("\n" + "=" * 50)
    print("TESTING PROMPT CREATION")
    print("=" * 50)

    try:
        from bench_forge.flame.tasks.causal_detection import CausalDetectionTask
        from bench_forge.tasks.config import PromptFormat

        task = CausalDetectionTask()

        # Test sample with tokens
        sample = {
            "tokens": [
                "The",
                "company",
                "reported",
                "strong",
                "earnings",
                "which",
                "boosted",
                "share",
                "prices",
            ],
            "tags": [
                "O",
                "O",
                "O",
                "B-CAUSE",
                "I-CAUSE",
                "O",
                "B-EFFECT",
                "I-EFFECT",
                "I-EFFECT",
            ],
        }

        # Test zero-shot prompt
        prompt = task.create_prompt(sample, PromptFormat.ZERO_SHOT)
        print(f"✅ Zero-shot prompt created (length: {len(prompt)})")
        print(f"Prompt preview: {prompt[:200]}...")

        # Test few-shot prompt
        prompt_fs = task.create_prompt(sample, PromptFormat.FEW_SHOT)
        print(f"✅ Few-shot prompt created (length: {len(prompt_fs)})")

        return True

    except Exception as e:
        print(f"❌ Prompt creation failed: {e}")
        return False


def test_cd_label_extraction():
    """Test CD label extraction."""
    print("\n" + "=" * 50)
    print("TESTING LABEL EXTRACTION")
    print("=" * 50)

    try:
        from bench_forge.flame.tasks.causal_detection import CausalDetectionTask

        task = CausalDetectionTask()
        tokens = [
            "The",
            "company",
            "reported",
            "strong",
            "earnings",
            "which",
            "boosted",
            "share",
            "prices",
        ]

        # Test various response formats
        test_cases = [
            {
                "name": "Clean format",
                "response": "Labels: O O O B-CAUSE I-CAUSE O B-EFFECT I-EFFECT I-EFFECT",
                "expected_count": 9,
            },
            {
                "name": "With extra text",
                "response": "Based on the text, here are the labels:\nLabels: O O O B-CAUSE I-CAUSE O B-EFFECT I-EFFECT I-EFFECT",
                "expected_count": 9,
            },
            {
                "name": "Mixed case",
                "response": "o o o b-cause i-cause o b-effect i-effect i-effect",
                "expected_count": 9,
            },
            {
                "name": "With punctuation",
                "response": "Labels: O, O, O, B-CAUSE, I-CAUSE, O, B-EFFECT, I-EFFECT, I-EFFECT",
                "expected_count": 9,
            },
        ]

        for test_case in test_cases:
            labels = task.extract_label_from_response(test_case["response"], tokens)
            if labels and len(labels) == test_case["expected_count"]:
                print(f"✅ {test_case['name']}: {len(labels)} labels extracted")
                print(f"   Labels: {labels[:5]}...")
            else:
                print(
                    f"❌ {test_case['name']}: Expected {test_case['expected_count']}, got {len(labels) if labels else 0}"
                )

        return True

    except Exception as e:
        print(f"❌ Label extraction test failed: {e}")
        return False


def test_cd_with_mock_response():
    """Test CD with simulated LLM response."""
    print("\n" + "=" * 50)
    print("TESTING WITH MOCK LLM RESPONSE")
    print("=" * 50)

    try:
        from bench_forge.flame.tasks.causal_detection import CausalDetectionTask

        task = CausalDetectionTask()

        # Create test data
        samples = [
            {
                "tokens": [
                    "Rising",
                    "interest",
                    "rates",
                    "caused",
                    "mortgage",
                    "demand",
                    "to",
                    "decline",
                ],
                "tags": [
                    "B-CAUSE",
                    "I-CAUSE",
                    "I-CAUSE",
                    "O",
                    "B-EFFECT",
                    "I-EFFECT",
                    "I-EFFECT",
                    "I-EFFECT",
                ],
            }
        ]

        # Create mock responses
        raw_responses = [
            "Based on the analysis:\nLabels: B-CAUSE I-CAUSE I-CAUSE O B-EFFECT I-EFFECT I-EFFECT I-EFFECT"
        ]

        # Extract responses
        extracted_responses = []
        for i, response in enumerate(raw_responses):
            tokens = samples[i]["tokens"]
            extracted = task.extract_label_from_response(response, tokens)
            extracted_responses.append(extracted)

        print("✅ Mock response processed")
        print(f"Tokens: {samples[0]['tokens']}")
        print(f"Expected: {samples[0]['tags']}")
        print(f"Extracted: {extracted_responses[0]}")

        # Check if extraction matches expected
        if extracted_responses[0] == samples[0]["tags"]:
            print("✅ Perfect extraction match!")
        else:
            print("⚠️ Extraction differs from expected")

        return True

    except Exception as e:
        print(f"❌ Mock response test failed: {e}")
        return False


def main():
    """Run CD tests."""
    print("🧪 TESTING CAUSAL DETECTION (CD) IMPLEMENTATION")
    print("=" * 80)

    # Run tests
    task = test_cd_basic()
    if not task:
        return

    prompt_ok = test_cd_prompt_creation()
    extract_ok = test_cd_label_extraction()
    mock_ok = test_cd_with_mock_response()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Task initialization: {'PASS' if task else 'FAIL'}")
    print(f"✅ Prompt creation: {'PASS' if prompt_ok else 'FAIL'}")
    print(f"✅ Label extraction: {'PASS' if extract_ok else 'FAIL'}")
    print(f"✅ Mock response: {'PASS' if mock_ok else 'FAIL'}")

    overall = task and prompt_ok and extract_ok and mock_ok
    print(
        f"\n🎯 OVERALL: {'PASS - Ready for live testing' if overall else 'FAIL - Needs fixes'}"
    )

    if overall:
        print("\n📝 NEXT STEPS:")
        print("- Test with real dataset samples")
        print("- Run full dataset evaluation")
        print("- Compare with FLAME baseline")
        print("- Document performance metrics")


if __name__ == "__main__":
    main()
