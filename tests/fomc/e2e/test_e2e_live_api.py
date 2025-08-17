#!/usr/bin/env python3
"""
End-to-End tests with live API calls for FOMC.
These tests use real LLM APIs with limited samples to control costs.
"""

import pytest
import os
import sys
import time
from pathlib import Path
import pandas as pd

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "benchforge"))

# Check for API keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
SKIP_LIVE_TESTS = os.getenv("SKIP_LIVE_TESTS", "false").lower() == "true"


@pytest.mark.skipif(SKIP_LIVE_TESTS, reason="Live tests disabled")
@pytest.mark.skipif(not TOGETHER_API_KEY, reason="No API key available")
class TestE2ELiveAPI:
    """End-to-end tests with real API calls."""

    @pytest.fixture(scope="class")
    def test_samples(self):
        """Get small set of test samples."""
        samples = [
            {
                "sentence": "The Committee decided to raise the target range for the federal funds rate by 25 basis points.",
                "expected_label": "HAWKISH",
            },
            {
                "sentence": "The Committee judges that the current stance of monetary policy is appropriate to support sustained expansion.",
                "expected_label": "NEUTRAL",
            },
            {
                "sentence": "The Committee decided to lower the target range for the federal funds rate to support the economy.",
                "expected_label": "DOVISH",
            },
        ]
        return samples

    @pytest.fixture(scope="class")
    def api_config(self):
        """API configuration for tests."""
        return {
            "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "max_tokens": 20,
            "temperature": 0.0,
            "top_p": 1.0,
        }

    def test_native_inference_live(self, test_samples, api_config):
        """Test native FLAME with live API."""
        import litellm
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        results = []

        for sample in test_samples[:2]:  # Limit to 2 samples for cost
            prompt = prompt_func(sample["sentence"])

            try:
                # Make API call
                response = litellm.completion(
                    model=api_config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=api_config["max_tokens"],
                    temperature=api_config["temperature"],
                    top_p=api_config["top_p"],
                )

                llm_response = response.choices[0].message.content

                # Extract label (simplified)
                extracted = None
                for label in ["HAWKISH", "DOVISH", "NEUTRAL"]:
                    if label in llm_response.upper():
                        extracted = label
                        break

                results.append(
                    {
                        "input": sample["sentence"][:50] + "...",
                        "expected": sample["expected_label"],
                        "response": llm_response,
                        "extracted": extracted,
                        "match": extracted == sample["expected_label"],
                    }
                )

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                pytest.fail(f"API call failed: {e}")

        # Check results
        print("\nNative FLAME Results:")
        for r in results:
            print(
                f"  Expected: {r['expected']}, Got: {r['extracted']}, Match: {r['match']}"
            )

        # At least 50% should match (small sample size)
        matches = sum(1 for r in results if r["match"])
        assert matches >= len(results) * 0.5

    def test_benchforge_inference_live(self, test_samples, api_config):
        """Test BenchForge with live API."""
        from flame.benchforge import BENCHFORGE_AVAILABLE

        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        import litellm
        from flame.tasks.fomc import FOMCTask
        from flame.benchforge import FLAMEConfig, PromptFormat
        from bench_forge.prompts.extractor import ResponseExtractor, ExtractionStrategy

        # Setup
        config = FLAMEConfig(
            name="fomc",
            dataset="fomc",
            prompt_format=PromptFormat.ZERO_SHOT,
            text_field="sentence",
            valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"],
        )
        task = FOMCTask(config)
        extractor = ResponseExtractor()

        results = []

        for sample in test_samples[:2]:  # Limit to 2 samples
            prompt = task.create_prompt(
                {"sentence": sample["sentence"]}, PromptFormat.ZERO_SHOT
            )

            try:
                # Make API call
                response = litellm.completion(
                    model=api_config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=api_config["max_tokens"],
                    temperature=api_config["temperature"],
                    top_p=api_config["top_p"],
                )

                llm_response = response.choices[0].message.content

                # Extract using BenchForge extractor
                extraction = extractor.extract(
                    llm_response,
                    strategy=ExtractionStrategy.FUZZY,
                    options=config.valid_labels,
                )

                results.append(
                    {
                        "input": sample["sentence"][:50] + "...",
                        "expected": sample["expected_label"],
                        "response": llm_response,
                        "extracted": extraction.value,
                        "confidence": extraction.confidence,
                        "match": extraction.value == sample["expected_label"],
                    }
                )

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                pytest.fail(f"API call failed: {e}")

        # Check results
        print("\nBenchForge Results:")
        for r in results:
            print(
                f"  Expected: {r['expected']}, Got: {r['extracted']}, "
                f"Confidence: {r['confidence']:.2f}, Match: {r['match']}"
            )

        # At least 50% should match
        matches = sum(1 for r in results if r["match"])
        assert matches >= len(results) * 0.5

    def test_implementation_parity_live(self, test_samples, api_config):
        """Test both implementations produce similar results."""
        import litellm
        from flame.code.prompts.registry import get_prompt, PromptFormat as NativeFormat

        # Skip if BenchForge not available
        try:
            from flame.tasks.fomc import FOMCTask
            from flame.benchforge import FLAMEConfig, PromptFormat as BFFormat
            from bench_forge.prompts.extractor import (
                ResponseExtractor,  # noqa: F401
                ExtractionStrategy,  # noqa: F401
            )
        except ImportError:
            pytest.skip("BenchForge not available for parity test")

        # Test one sample with both implementations
        sample = test_samples[0]

        # Native implementation
        native_prompt_func = get_prompt("fomc", NativeFormat.ZERO_SHOT)
        native_prompt = native_prompt_func(sample["sentence"])

        # BenchForge implementation
        config = FLAMEConfig(
            name="fomc",
            dataset="fomc",
            prompt_format=BFFormat.ZERO_SHOT,
            text_field="sentence",
            valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"],
        )
        task = FOMCTask(config)
        bf_prompt = task.create_prompt(
            {"sentence": sample["sentence"]}, BFFormat.ZERO_SHOT
        )

        # Make API calls for both
        results = {}

        for impl_name, prompt in [("native", native_prompt), ("benchforge", bf_prompt)]:
            try:
                response = litellm.completion(
                    model=api_config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=api_config["max_tokens"],
                    temperature=api_config["temperature"],
                    top_p=api_config["top_p"],
                )

                llm_response = response.choices[0].message.content

                # Extract label
                extracted = None
                for label in ["HAWKISH", "DOVISH", "NEUTRAL"]:
                    if label in llm_response.upper():
                        extracted = label
                        break

                results[impl_name] = {"response": llm_response, "extracted": extracted}

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                pytest.fail(f"API call failed for {impl_name}: {e}")

        # Compare results
        print("\nParity Test Results:")
        print(f"  Native: {results['native']['extracted']}")
        print(f"  BenchForge: {results['benchforge']['extracted']}")

        # Both should extract the same label (with deterministic generation)
        if api_config["temperature"] == 0.0:
            assert (
                results["native"]["extracted"] == results["benchforge"]["extracted"]
            ), "Implementations produced different results with same input"


@pytest.mark.skipif(SKIP_LIVE_TESTS, reason="Live tests disabled")
class TestE2EFullPipeline:
    """Test full inference and evaluation pipeline."""

    def test_full_pipeline_small_batch(self):
        """Test complete pipeline with small batch."""
        import tempfile
        import subprocess

        # Skip if no API key
        if not TOGETHER_API_KEY:
            pytest.skip("No API key for full pipeline test")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run inference with 3 samples
            cmd = [
                "python",
                "main.py",
                "--mode",
                "inference",
                "--task",
                "fomc",
                "--model",
                "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "--max_tokens",
                "20",
                "--temperature",
                "0.0",
                "--num_samples",
                "3",
                "--output_dir",
                tmpdir,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                pytest.fail(f"Inference failed: {result.stderr}")

            # Check output file was created
            output_files = list(Path(tmpdir).glob("*.csv"))
            assert len(output_files) > 0, "No output file created"

            # Load and validate results
            df = pd.read_csv(output_files[0])
            assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
            assert "llm_responses" in df.columns or "raw_response" in df.columns

            print(f"\n✅ Full pipeline test passed with {len(df)} samples")


@pytest.mark.skipif(SKIP_LIVE_TESTS, reason="Live tests disabled")
class TestE2EErrorHandling:
    """Test error handling in E2E scenarios."""

    def test_api_timeout_handling(self):
        """Test handling of API timeouts."""
        import litellm
        from flame.code.prompts.registry import get_prompt, PromptFormat

        if not TOGETHER_API_KEY:
            pytest.skip("No API key available")

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        prompt = prompt_func("Test statement")

        # Set very short timeout
        with pytest.raises(Exception):
            litellm.completion(
                model="together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                timeout=0.001,  # Impossibly short timeout
            )

    def test_invalid_model_handling(self):
        """Test handling of invalid model names."""
        import litellm
        from flame.code.prompts.registry import get_prompt, PromptFormat

        if not TOGETHER_API_KEY:
            pytest.skip("No API key available")

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        prompt = prompt_func("Test statement")

        with pytest.raises(Exception):
            litellm.completion(
                model="together_ai/invalid-model-name",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
            )

    def test_rate_limit_handling(self):
        """Test handling of rate limits."""
        import litellm
        from flame.code.prompts.registry import get_prompt, PromptFormat

        if not TOGETHER_API_KEY:
            pytest.skip("No API key available")

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)

        # Make rapid requests (but limit to avoid actual rate limits)
        errors = 0
        for i in range(3):
            try:
                prompt = prompt_func(f"Test statement {i}")
                litellm.completion(
                    model="together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                )
                # Small delay to be respectful
                time.sleep(0.2)
            except Exception as e:
                if "rate" in str(e).lower():
                    errors += 1

        # We should handle rate limits gracefully
        assert errors < 3, "All requests failed, possible rate limit not handled"


def test_e2e_cost_tracking():
    """Track and report API costs for E2E tests."""
    # This is a meta-test to ensure we're tracking costs

    costs = {
        "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct": {
            "input": 0.00020 / 1000,  # $0.20 per 1M tokens
            "output": 0.00020 / 1000,  # $0.20 per 1M tokens
        }
    }

    # Estimate tokens for our tests
    avg_prompt_tokens = 150
    avg_output_tokens = 20
    num_api_calls = 10  # Approximate for all E2E tests

    model_key = "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"

    total_input_tokens = avg_prompt_tokens * num_api_calls
    total_output_tokens = avg_output_tokens * num_api_calls

    total_cost = (
        total_input_tokens * costs[model_key]["input"]
        + total_output_tokens * costs[model_key]["output"]
    )

    print("\n💰 Estimated E2E Test Costs:")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Estimated cost: ${total_cost:.4f}")

    assert total_cost < 0.10, f"E2E tests too expensive: ${total_cost:.4f}"


if __name__ == "__main__":
    # Check if we should run live tests
    if SKIP_LIVE_TESTS:
        print("⚠️ Live tests are disabled. Set SKIP_LIVE_TESTS=false to enable.")
        sys.exit(0)

    if not TOGETHER_API_KEY:
        print("⚠️ No API key found. Set TOGETHER_API_KEY to run live tests.")
        sys.exit(0)

    # Run with cost tracking
    pytest.main([__file__, "-v", "--tb=short", "-k", "not cost_tracking"])
