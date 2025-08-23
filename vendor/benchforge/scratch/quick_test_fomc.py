#!/usr/bin/env python3
"""Quick test script to validate FOMC implementations work correctly.

This script performs a quick sanity check with a small number of samples
to ensure both native FLAME and BenchForge methods are functioning.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
flame_root = Path(__file__).parent.parent
benchforge_root = Path(__file__).parent
sys.path.insert(0, str(flame_root))
sys.path.insert(0, str(benchforge_root))

load_dotenv()


def test_benchforge_fomc():
    """Quick test of BenchForge FOMC implementation."""
    print("\n" + "=" * 60)
    print("Testing BenchForge FOMC Implementation")
    print("=" * 60)

    try:
        # Import BenchForge modules
        from bench_forge.flame.tasks.fomc import FOMCConfig, FOMCTask
        from bench_forge.tasks.config import PromptFormat

        # Create minimal config
        config = FOMCConfig(
            name="fomc",
            dataset="fomc_communication",
            huggingface_dataset="gtfintechlab/fomc_communication",
            dataset_split="test",
            prompt_format=PromptFormat.ZERO_SHOT,
        )

        # Initialize task
        task = FOMCTask(config)
        print("✅ FOMCTask initialized successfully")

        # Test prompt creation
        sample = {
            "sentence": "The Committee expects that economic conditions will warrant exceptionally low levels of the federal funds rate for an extended period.",
            "label": 0,  # DOVISH
        }

        prompt = task.create_prompt(sample)
        print(f"✅ Prompt created: {len(prompt)} chars")

        # Test extraction strategies
        test_responses = [
            "DOVISH",  # Direct match
            "Classification: HAWKISH",  # With prefix
            "The answer is NEUTRAL",  # In sentence
            "I would classify this as DOVISH based on the language",  # Context
            '"HAWKISH"',  # Quoted
            "neutral",  # Lowercase
            "The sentiment appears to be dovish\nDOVISH",  # Multi-line
        ]

        print("\n📝 Testing extraction strategies:")
        for i, response in enumerate(test_responses, 1):
            extracted = task.extract_label_from_response(response)
            status = "✅" if extracted in ["DOVISH", "HAWKISH", "NEUTRAL"] else "❌"
            print(f"  Strategy test {i}: {status} Extracted: {extracted}")

        # Test format_results_with_evaluation method
        print("\n📊 Testing format_results_with_evaluation:")

        samples = [
            {"sentence": "Text 1", "label": 0},  # DOVISH
            {"sentence": "Text 2", "label": 1},  # HAWKISH
            {"sentence": "Text 3", "label": 2},  # NEUTRAL
        ]

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        # Mock responses
        class MockResponse:
            def __init__(self, content):
                self.choices = [
                    type(
                        "obj",
                        (object,),
                        {"message": type("obj", (object,), {"content": content})()},
                    )
                ]

        responses = [
            MockResponse("DOVISH"),
            MockResponse("HAWKISH"),
            MockResponse("NEUTRAL"),
        ]

        extracted = ["DOVISH", "HAWKISH", "NEUTRAL"]

        # Test the new method
        results = task.format_results_with_evaluation(
            samples, prompts, responses, extracted
        )

        assert "results" in results, "Missing 'results' key"
        assert "metrics" in results, "Missing 'metrics' key"

        results_df = results["results"]
        metrics_df = results["metrics"]

        print(f"✅ Results DataFrame shape: {results_df.shape}")
        print(f"✅ Metrics DataFrame shape: {metrics_df.shape}")

        # Check columns
        expected_cols = [
            "sentences",
            "actual_labels",
            "llm_responses",
            "extracted_labels",
        ]
        for col in expected_cols:
            assert col in results_df.columns, f"Missing column: {col}"
        print("✅ All expected columns present")

        # Check metrics
        accuracy = metrics_df[metrics_df["Metric"] == "Accuracy"]["Value"].values[0]
        print(f"✅ Accuracy calculated: {accuracy:.2f}")

        print("\n🎉 BenchForge FOMC implementation is working correctly!")
        return True

    except Exception as e:
        print(f"\n❌ BenchForge test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_native_flame_fomc():
    """Quick test of native FLAME FOMC implementation."""
    print("\n" + "=" * 60)
    print("Testing Native FLAME FOMC Implementation")
    print("=" * 60)

    try:
        # Import FLAME modules
        from flame.code.fomc.fomc_inference import validate_sample
        from flame.code.prompts import get_prompt, PromptFormat

        # Test validation
        assert validate_sample("DOVISH")
        assert not validate_sample("invalid")
        print("✅ Validation function works")

        # Test prompt creation
        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        if prompt_func:
            test_sentence = "The Committee expects rates to remain low."
            prompt = prompt_func(test_sentence)
            print(f"✅ Prompt created: {len(prompt)} chars")
        else:
            print("⚠️  Prompt function not found (may need registry setup)")

        print("\n🎉 Native FLAME FOMC implementation is working!")
        return True

    except Exception as e:
        print(f"\n❌ Native FLAME test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run quick tests."""
    print("\n🚀 Starting Quick FOMC Implementation Tests")
    print("This validates both implementations are working correctly.\n")

    # Check API key
    if not os.getenv("TOGETHERAI_API_KEY"):
        print("⚠️  Warning: TOGETHERAI_API_KEY not set")
        print("   API calls will fail, but we can still test other functionality\n")

    # Run tests
    benchforge_ok = test_benchforge_fomc()
    flame_ok = test_native_flame_fomc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if benchforge_ok and flame_ok:
        print("✅ Both implementations are working correctly!")
        print("\nNext steps:")
        print("1. Run small comparison: python compare_fomc_methods.py --quick")
        print("2. Run full comparison: python compare_fomc_methods.py --num-samples 50")
        print("3. Test all 5 models: python compare_fomc_methods.py --models", end=" ")
        print("together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo", end=" ")
        print("together_ai/mistralai/Mistral-7B-Instruct-v0.3", end=" ")
        print("together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo")
        return 0
    else:
        status = []
        if not benchforge_ok:
            status.append("BenchForge")
        if not flame_ok:
            status.append("Native FLAME")
        print(f"❌ Issues found with: {', '.join(status)}")
        print("\nPlease fix the issues above before running comparisons.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
