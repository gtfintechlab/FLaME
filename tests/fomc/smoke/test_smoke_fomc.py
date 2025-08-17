#!/usr/bin/env python3
"""
Smoke tests for FOMC implementations.
Quick sanity checks that can run in <10 seconds.
"""

import pytest
import sys
import time
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "benchforge"))


class TestSmoke:
    """Quick smoke tests for both implementations."""

    def test_native_imports(self):
        """Test native FLAME imports work."""
        try:
            from flame.code.prompts.registry import get_prompt, PromptFormat  # noqa: F401
            from flame.code.fomc.fomc_evaluate import map_label_to_number  # noqa: F401

            assert True
        except ImportError as e:
            pytest.fail(f"Native imports failed: {e}")

    def test_benchforge_imports(self):
        """Test BenchForge imports work."""
        try:
            from flame.benchforge import BENCHFORGE_AVAILABLE

            if BENCHFORGE_AVAILABLE:
                from flame.tasks.fomc import FOMCTask  # noqa: F401
                from flame.benchforge import FLAMEConfig, PromptFormat  # noqa: F401

                assert True
            else:
                pytest.skip("BenchForge not available")
        except ImportError as e:
            pytest.fail(f"BenchForge imports failed: {e}")

    def test_native_prompt_generation_quick(self):
        """Quick test of native prompt generation."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        start = time.time()
        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        prompt = prompt_func("Fed raised rates")
        elapsed = time.time() - start

        assert prompt is not None
        assert len(prompt) > 100
        assert elapsed < 1.0  # Should be very fast

    def test_benchforge_task_creation_quick(self):
        """Quick test of BenchForge task creation."""
        from flame.benchforge import BENCHFORGE_AVAILABLE

        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.tasks.fomc import FOMCTask
        from flame.benchforge import FLAMEConfig, PromptFormat

        start = time.time()
        config = FLAMEConfig(
            name="fomc", dataset="fomc", prompt_format=PromptFormat.ZERO_SHOT
        )
        task = FOMCTask(config)
        elapsed = time.time() - start

        assert task is not None
        assert elapsed < 1.0  # Should be fast

    def test_label_mapping_quick(self):
        """Quick test of label mapping."""
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        assert map_label_to_number("HAWKISH") == 1
        assert map_label_to_number("DOVISH") == 0
        assert map_label_to_number("NEUTRAL") == 2

    def test_critical_path_native(self):
        """Test critical path for native implementation."""
        from flame.code.prompts.registry import get_prompt, PromptFormat
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        # Simulate critical path
        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        prompt = prompt_func("The Fed maintained rates")

        # Simulate extraction
        extracted = "NEUTRAL"  # Simplified extraction

        # Map to number
        label_num = map_label_to_number(extracted)

        assert prompt is not None
        assert label_num == 2

    def test_critical_path_benchforge(self):
        """Test critical path for BenchForge implementation."""
        from flame.benchforge import BENCHFORGE_AVAILABLE

        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.tasks.fomc import FOMCTask
        from flame.benchforge import FLAMEConfig, PromptFormat

        # Create task
        config = FLAMEConfig(
            name="fomc",
            dataset="fomc",
            prompt_format=PromptFormat.ZERO_SHOT,
            text_field="sentence",
            valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"],
        )
        task = FOMCTask(config)

        # Generate prompt
        sample = {"sentence": "The Fed maintained rates"}
        prompt = task.create_prompt(sample, PromptFormat.ZERO_SHOT)

        assert prompt is not None
        assert "HAWKISH" in prompt
        assert "DOVISH" in prompt
        assert "NEUTRAL" in prompt


class TestSmokePerformance:
    """Performance smoke tests."""

    def test_prompt_generation_under_100ms(self):
        """Test prompt generation is under 100ms."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)

        # Warm up
        prompt_func("warm up")

        # Time 10 generations
        start = time.time()
        for _ in range(10):
            prompt_func("Test statement")
        elapsed = time.time() - start

        avg_time = elapsed / 10
        assert avg_time < 0.1, f"Prompt generation too slow: {avg_time * 1000:.2f}ms"

    def test_label_extraction_under_10ms(self):
        """Test label extraction is under 10ms."""
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        labels = ["HAWKISH", "DOVISH", "NEUTRAL"] * 10

        start = time.time()
        for label in labels:
            map_label_to_number(label)
        elapsed = time.time() - start

        avg_time = elapsed / 30
        assert avg_time < 0.01, f"Label extraction too slow: {avg_time * 1000:.2f}ms"


class TestSmokeParity:
    """Quick parity checks between implementations."""

    def test_both_support_same_labels(self):
        """Test both implementations support same label set."""
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        expected_labels = {"HAWKISH": 1, "DOVISH": 0, "NEUTRAL": 2}

        for label, expected_num in expected_labels.items():
            assert map_label_to_number(label) == expected_num

        # If BenchForge available, test it too
        try:
            from flame.benchforge import BENCHFORGE_AVAILABLE

            if BENCHFORGE_AVAILABLE:
                from flame.tasks.fomc import FOMCTask
                from flame.benchforge import FLAMEConfig

                config = FLAMEConfig(
                    name="fomc", valid_labels=list(expected_labels.keys())
                )
                FOMCTask(config)

                # Check labels match
                assert set(config.valid_labels) == set(expected_labels.keys())
        except ImportError:
            pass  # BenchForge not required for smoke test

    def test_prompt_contains_required_elements(self):
        """Test prompts contain required elements."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        prompt = prompt_func("Test input")

        required_elements = [
            "HAWKISH",
            "DOVISH",
            "NEUTRAL",
            "Test input",  # Input should be in prompt
        ]

        for element in required_elements:
            assert element in prompt, f"Missing required element: {element}"


def test_smoke_suite_completes_quickly():
    """Meta-test: Ensure smoke suite runs quickly."""
    start = time.time()

    # Run a few quick checks
    test = TestSmoke()
    test.test_native_imports()
    test.test_label_mapping_quick()

    elapsed = time.time() - start
    assert elapsed < 1.0, f"Smoke tests too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    # Run with timing
    import subprocess
    import sys

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    print(f"\n⏱️ Smoke tests completed in {elapsed:.2f} seconds")

    if elapsed > 10:
        print("⚠️ WARNING: Smoke tests took longer than 10 seconds!")
        sys.exit(1)

    sys.exit(result.returncode)
