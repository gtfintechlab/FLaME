#!/usr/bin/env python3
"""
Unit tests for FOMC prompt generation functions.
Tests both native FLAME and BenchForge implementations.
"""

import pytest
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "benchforge"))


class TestNativePromptGeneration:
    """Test native FLAME prompt generation."""

    def test_zero_shot_prompt_structure(self):
        """Test zero-shot prompt has correct structure."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        test_sentence = "The Federal Reserve raised interest rates."
        prompt = prompt_func(test_sentence)

        # Check required elements
        assert "HAWKISH" in prompt
        assert "DOVISH" in prompt
        assert "NEUTRAL" in prompt
        assert test_sentence in prompt
        assert "Classify" in prompt or "classify" in prompt

    def test_prompt_with_empty_input(self):
        """Test prompt generation with empty input."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        prompt = prompt_func("")

        # Should still generate valid prompt structure
        assert "HAWKISH" in prompt
        assert "DOVISH" in prompt
        assert "NEUTRAL" in prompt

    def test_prompt_with_special_characters(self):
        """Test prompt generation with special characters."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        test_sentence = 'The Fed\'s decision: "raise rates by 0.25%" - unanimous.'
        prompt = prompt_func(test_sentence)

        assert test_sentence in prompt
        assert prompt.count('"') >= 2  # Original quotes preserved

    def test_prompt_consistency(self):
        """Test that same input produces same prompt."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        test_sentence = "The Committee decided to maintain the target range."

        prompt1 = prompt_func(test_sentence)
        prompt2 = prompt_func(test_sentence)

        assert prompt1 == prompt2

    def test_prompt_label_order(self):
        """Test that labels appear in consistent order."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        prompt = prompt_func("Test statement")

        # Find positions of labels
        hawkish_pos = prompt.find("HAWKISH")
        dovish_pos = prompt.find("DOVISH")
        neutral_pos = prompt.find("NEUTRAL")

        assert hawkish_pos > -1
        assert dovish_pos > -1
        assert neutral_pos > -1
        # Labels should appear in some order
        assert hawkish_pos != dovish_pos != neutral_pos


class TestBenchForgePromptGeneration:
    """Test BenchForge prompt generation."""

    @pytest.fixture
    def fomc_task(self):
        """Create FOMCTask instance for testing."""
        from flame.benchforge import BENCHFORGE_AVAILABLE

        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.tasks.fomc import FOMCTask
        from flame.benchforge import FLAMEConfig, PromptFormat

        config = FLAMEConfig(
            name="fomc",
            dataset="fomc",
            prompt_format=PromptFormat.ZERO_SHOT,
            text_field="sentence",
            label_field="label",
            valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"],
        )
        return FOMCTask(config)

    def test_benchforge_prompt_structure(self, fomc_task):
        """Test BenchForge prompt has correct structure."""
        from flame.benchforge import PromptFormat

        test_sample = {"sentence": "The Federal Reserve raised interest rates."}
        prompt = fomc_task.create_prompt(test_sample, PromptFormat.ZERO_SHOT)

        # Check required elements
        assert "HAWKISH" in prompt
        assert "DOVISH" in prompt
        assert "NEUTRAL" in prompt
        assert test_sample["sentence"] in prompt

    def test_benchforge_prompt_consistency(self, fomc_task):
        """Test BenchForge produces consistent prompts."""
        from flame.benchforge import PromptFormat

        test_sample = {"sentence": "The Committee decided to maintain rates."}

        prompt1 = fomc_task.create_prompt(test_sample, PromptFormat.ZERO_SHOT)
        prompt2 = fomc_task.create_prompt(test_sample, PromptFormat.ZERO_SHOT)

        assert prompt1 == prompt2

    def test_benchforge_handles_missing_field(self, fomc_task):
        """Test BenchForge handles missing text field gracefully."""
        from flame.benchforge import PromptFormat

        # Missing 'sentence' field
        test_sample = {"text": "Wrong field name"}

        with pytest.raises((KeyError, AttributeError)):
            fomc_task.create_prompt(test_sample, PromptFormat.ZERO_SHOT)

    def test_benchforge_prompt_format_types(self, fomc_task):
        """Test different prompt format types."""
        from flame.benchforge import PromptFormat

        test_sample = {"sentence": "Test statement"}

        # Should support zero-shot
        zero_shot = fomc_task.create_prompt(test_sample, PromptFormat.ZERO_SHOT)
        assert zero_shot is not None
        assert len(zero_shot) > 0

        # Few-shot might not be implemented yet
        try:
            few_shot = fomc_task.create_prompt(test_sample, PromptFormat.FEW_SHOT)
            assert few_shot != zero_shot  # Should be different
        except (NotImplementedError, AttributeError):
            pass  # Few-shot not required for Phase 1


class TestPromptCompatibility:
    """Test compatibility between native and BenchForge prompts."""

    def test_both_contain_same_labels(self):
        """Test both implementations use same label set."""
        from flame.code.prompts.registry import get_prompt, PromptFormat as NativeFormat

        try:
            from flame.tasks.fomc import FOMCTask
            from flame.benchforge import FLAMEConfig, PromptFormat as BFFormat

            benchforge_available = True
        except ImportError:
            benchforge_available = False

        # Native prompt
        native_func = get_prompt("fomc", NativeFormat.ZERO_SHOT)
        native_prompt = native_func("Test statement")

        if benchforge_available:
            # BenchForge prompt
            config = FLAMEConfig(
                name="fomc",
                dataset="fomc",
                prompt_format=BFFormat.ZERO_SHOT,
                text_field="sentence",
                label_field="label",
                valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"],
            )
            task = FOMCTask(config)
            bf_prompt = task.create_prompt(
                {"sentence": "Test statement"}, BFFormat.ZERO_SHOT
            )

            # Both should contain all three labels
            for label in ["HAWKISH", "DOVISH", "NEUTRAL"]:
                assert label in native_prompt
                assert label in bf_prompt

    def test_prompt_length_reasonable(self):
        """Test prompt lengths are reasonable."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)

        short_input = "Fed raised rates."
        long_input = "The Federal Open Market Committee decided to raise the target range for the federal funds rate by 25 basis points to 5.25-5.50 percent, citing persistent inflationary pressures and strong labor market conditions that warrant a more restrictive monetary policy stance to bring inflation back to the Committee's 2 percent objective over time."

        short_prompt = prompt_func(short_input)
        long_prompt = prompt_func(long_input)

        # Prompts should be reasonable length
        assert 100 < len(short_prompt) < 1000
        assert 200 < len(long_prompt) < 2000

        # Longer input should produce longer prompt
        assert len(long_prompt) > len(short_prompt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
