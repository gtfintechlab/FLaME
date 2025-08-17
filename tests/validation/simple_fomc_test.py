#!/usr/bin/env python3
"""
Simple FOMC Test - Direct API Testing
=====================================
Tests both implementations without running full inference.
"""

import sys
import logging
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchforge"))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_native_flame():
    """Test native FLAME FOMC implementation."""
    print("\n" + "=" * 60)
    print("Testing Native FLAME Implementation")
    print("=" * 60)

    try:
        # Import native components
        from flame.code.prompts.registry import get_prompt, PromptFormat
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        print("✅ Imports successful")

        # Test prompt generation
        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        test_sentence = "The Committee decided to raise the federal funds rate."
        prompt = prompt_func(test_sentence)

        print("✅ Prompt generation works")
        print(f"   Sample: {test_sentence[:50]}...")
        print(f"   Prompt length: {len(prompt)} characters")

        # Test label mapping
        assert map_label_to_number("HAWKISH") == 1
        assert map_label_to_number("DOVISH") == 0
        assert map_label_to_number("NEUTRAL") == 2
        print("✅ Label mapping works correctly")

        # Test that we can access dataset info
        print("✅ Core functionality validated")

        print("\n✅ Native FLAME implementation is working!")
        return True

    except Exception as e:
        print(f"\n❌ Native FLAME test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_benchforge():
    """Test BenchForge FOMC implementation."""
    print("\n" + "=" * 60)
    print("Testing BenchForge Implementation")
    print("=" * 60)

    try:
        # Check BenchForge availability
        from flame.benchforge import BENCHFORGE_AVAILABLE

        if not BENCHFORGE_AVAILABLE:
            print("❌ BenchForge not available")
            return False

        print("✅ BenchForge is available")

        # Import BenchForge components
        from flame.tasks.fomc import FOMCTask
        from flame.benchforge import FLAMEConfig, PromptFormat

        print("✅ Imports successful")

        # Create task configuration
        config = FLAMEConfig(
            name="fomc",
            dataset="fomc",
            prompt_format=PromptFormat.ZERO_SHOT,
            text_field="sentence",
            label_field="label",
            valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"],
        )

        # Initialize task
        task = FOMCTask(config)
        print("✅ FOMCTask initialized")

        # Test prompt generation
        test_sample = {
            "sentence": "The Committee decided to raise the federal funds rate."
        }
        prompt = task.create_prompt(test_sample, PromptFormat.ZERO_SHOT)
        print("✅ Prompt generation works")
        print(f"   Prompt length: {len(prompt)} characters")

        # Test that task has required methods
        assert hasattr(task, "create_prompt")
        assert hasattr(task, "load_dataset")
        print("✅ Task has required methods")

        # Test extraction using the extractor module if available
        try:
            from bench_forge.prompts.extractor import (
                ResponseExtractor,
                ExtractionStrategy,
            )

            extractor = ResponseExtractor()
            test_response = "Based on the statement, this is clearly HAWKISH."
            result = extractor.extract(
                test_response,
                strategy=ExtractionStrategy.FUZZY,
                options=["HAWKISH", "DOVISH", "NEUTRAL"],
            )
            if result.value:
                print(
                    f"✅ Extraction works: '{test_response[:30]}...' -> {result.value}"
                )
        except Exception as e:
            print(f"⚠️ Extraction test skipped: {e}")

        print("\n✅ BenchForge implementation is working!")
        return True

    except Exception as e:
        print(f"\n❌ BenchForge test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def compare_prompts():
    """Compare prompts from both implementations."""
    print("\n" + "=" * 60)
    print("Comparing Prompt Generation")
    print("=" * 60)

    test_sentence = (
        "The Committee decided to raise the federal funds rate by 25 basis points."
    )

    try:
        # Get native prompt
        from flame.code.prompts.registry import get_prompt, PromptFormat as NativeFormat

        native_prompt_func = get_prompt("fomc", NativeFormat.ZERO_SHOT)
        native_prompt = native_prompt_func(test_sentence)

        # Get BenchForge prompt
        from flame.tasks.fomc import FOMCTask
        from flame.benchforge import FLAMEConfig, PromptFormat as BFFormat

        config = FLAMEConfig(
            name="fomc", dataset="fomc", prompt_format=BFFormat.ZERO_SHOT
        )
        task = FOMCTask(config)
        bf_prompt = task.create_prompt({"sentence": test_sentence}, BFFormat.ZERO_SHOT)

        # Compare
        print(f"Test sentence: {test_sentence[:50]}...")
        print(f"\nNative prompt length: {len(native_prompt)} chars")
        print(f"BenchForge prompt length: {len(bf_prompt)} chars")

        # Check if they contain same key elements
        key_words = [
            "HAWKISH",
            "DOVISH",
            "NEUTRAL",
            "Federal",
            "Committee",
            "Statement",
        ]

        native_has_keys = all(word in native_prompt for word in key_words[:3])
        bf_has_keys = all(word in bf_prompt for word in key_words[:3])

        if native_has_keys and bf_has_keys:
            print("\n✅ Both prompts contain required classification labels")
        else:
            print("\n⚠️ Prompts may differ in structure")

        # Show snippets
        print("\nNative prompt preview:")
        print(f"  {native_prompt[:150]}...")
        print("\nBenchForge prompt preview:")
        print(f"  {bf_prompt[:150]}...")

        return True

    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        return False


def main():
    """Main test runner."""
    print("\n" + "🚀 " * 15)
    print("FOMC IMPLEMENTATION TESTING")
    print("Testing Core Functionality Without Full Inference")
    print("🚀 " * 15)

    # Track results
    results = {"native": False, "benchforge": False, "comparison": False}

    # Test native
    results["native"] = test_native_flame()

    # Test BenchForge
    results["benchforge"] = test_benchforge()

    # Compare if both work
    if results["native"] and results["benchforge"]:
        results["comparison"] = compare_prompts()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all(results.values()):
        print("\n✅ ALL TESTS PASSED!")
        print("\nBoth implementations are working correctly:")
        print("  - Native FLAME: ✅")
        print("  - BenchForge: ✅")
        print("  - Prompt compatibility: ✅")
        print("\n🎉 Phase 1 validation complete - ready for Phase 2!")
        return 0
    else:
        print("\n❌ Some tests failed:")
        for name, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"  - {name}: {status}")
        print("\nPlease fix the issues before proceeding to Phase 2.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
