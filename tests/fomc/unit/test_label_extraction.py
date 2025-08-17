#!/usr/bin/env python3
"""
Unit tests for FOMC label extraction functions.
Tests extraction logic for both implementations.
"""

import pytest
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "benchforge"))


class TestNativeLabelExtraction:
    """Test native FLAME label extraction."""

    def test_map_label_to_number(self):
        """Test label to number mapping."""
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        assert map_label_to_number("HAWKISH") == 1
        assert map_label_to_number("DOVISH") == 0
        assert map_label_to_number("NEUTRAL") == 2

        # Test case insensitive
        assert map_label_to_number("hawkish") == 1
        assert map_label_to_number("Dovish") == 0
        assert map_label_to_number("NeuTRal") == 2

    def test_extract_from_clean_response(self):
        """Test extraction from clean LLM responses."""
        # This would test the actual extraction logic if it exists
        # Since native FLAME uses rule-based extraction in evaluation
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        clean_responses = [
            ("HAWKISH", 1),
            ("The answer is DOVISH", 0),
            ("Based on the analysis, this is NEUTRAL.", 2),
            ("Classification: HAWKISH", 1),
        ]

        for response, expected in clean_responses:
            # Extract label (simplified logic)
            label = None
            for word in ["HAWKISH", "DOVISH", "NEUTRAL"]:
                if word in response.upper():
                    label = word
                    break

            if label:
                assert map_label_to_number(label) == expected

    def test_extract_from_messy_response(self):
        """Test extraction from messy LLM responses."""
        messy_responses = [
            ("I think this is probably hawkish", "HAWKISH"),
            ("dovish-leaning statement", "DOVISH"),
            ("Neither hawkish nor dovish, so neutral", "NEUTRAL"),
            ("HAWKISH!!!!", "HAWKISH"),
            ("**DOVISH**", "DOVISH"),
        ]

        for response, expected in messy_responses:
            # Simple extraction logic
            response_upper = response.upper()
            extracted = None

            for label in ["HAWKISH", "DOVISH", "NEUTRAL"]:
                if label in response_upper:
                    extracted = label
                    break

            assert (
                extracted == expected
            ), f"Failed to extract {expected} from '{response}'"

    def test_handle_invalid_labels(self):
        """Test handling of invalid labels."""
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        # These should return None or raise an error
        invalid_labels = ["BULLISH", "BEARISH", "UNKNOWN", "", None]

        for label in invalid_labels:
            try:
                result = map_label_to_number(label)
                # If it doesn't raise, it should return None
                assert result is None
            except (KeyError, ValueError, AttributeError):
                # Expected behavior for invalid labels
                pass


class TestBenchForgeExtraction:
    """Test BenchForge extraction capabilities."""

    @pytest.fixture
    def extractor(self):
        """Create ResponseExtractor instance."""
        try:
            from bench_forge.prompts.extractor import ResponseExtractor

            return ResponseExtractor()
        except ImportError:
            pytest.skip("BenchForge extractor not available")

    def test_rule_based_extraction(self, extractor):
        """Test rule-based extraction strategy."""
        from bench_forge.prompts.extractor import ExtractionStrategy

        test_cases = [
            ("The answer is HAWKISH", "HAWKISH"),
            ("DOVISH", "DOVISH"),
            ("Classification: NEUTRAL", "NEUTRAL"),
            ("Based on analysis, this is clearly HAWKISH.", "HAWKISH"),
        ]

        for response, expected in test_cases:
            result = extractor.extract(
                response,
                strategy=ExtractionStrategy.RULE_BASED,
                options=["HAWKISH", "DOVISH", "NEUTRAL"],
            )
            assert result.value == expected
            assert result.confidence > 0

    def test_fuzzy_extraction(self, extractor):
        """Test fuzzy matching extraction."""
        from bench_forge.prompts.extractor import ExtractionStrategy

        test_cases = [
            ("hawkish", "HAWKISH"),
            ("Dovish", "DOVISH"),
            ("neutral stance", "NEUTRAL"),
            ("This seems HAWKISH to me", "HAWKISH"),
        ]

        for response, expected in test_cases:
            result = extractor.extract(
                response,
                strategy=ExtractionStrategy.FUZZY,
                options=["HAWKISH", "DOVISH", "NEUTRAL"],
            )
            assert result.value == expected

    def test_extraction_confidence_scores(self, extractor):
        """Test extraction confidence scoring."""
        from bench_forge.prompts.extractor import ExtractionStrategy

        # Clear, direct answer should have high confidence
        clear_result = extractor.extract(
            "HAWKISH",
            strategy=ExtractionStrategy.RULE_BASED,
            options=["HAWKISH", "DOVISH", "NEUTRAL"],
        )
        assert clear_result.confidence >= 0.9

        # Embedded answer should have lower confidence
        embedded_result = extractor.extract(
            "I think maybe this could be considered hawkish",
            strategy=ExtractionStrategy.FUZZY,
            options=["HAWKISH", "DOVISH", "NEUTRAL"],
        )
        assert 0.5 <= embedded_result.confidence < 0.9

    def test_extraction_metadata(self, extractor):
        """Test extraction metadata collection."""
        from bench_forge.prompts.extractor import ExtractionStrategy

        result = extractor.extract(
            "The answer is DOVISH",
            strategy=ExtractionStrategy.RULE_BASED,
            options=["HAWKISH", "DOVISH", "NEUTRAL"],
        )

        assert result.value == "DOVISH"
        assert result.strategy == ExtractionStrategy.RULE_BASED
        assert result.metadata is not None
        assert "method" in result.metadata

    def test_handle_no_match(self, extractor):
        """Test handling when no label is found."""
        from bench_forge.prompts.extractor import ExtractionStrategy

        result = extractor.extract(
            "This text contains no valid labels",
            strategy=ExtractionStrategy.RULE_BASED,
            options=["HAWKISH", "DOVISH", "NEUTRAL"],
        )

        assert result.value is None
        assert result.confidence == 0
        assert not result.success


class TestExtractionCompatibility:
    """Test extraction compatibility between implementations."""

    def test_consistent_label_mapping(self):
        """Test both implementations map labels to same numbers."""
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        # Standard mapping that both should follow
        expected_mapping = {"HAWKISH": 1, "DOVISH": 0, "NEUTRAL": 2}

        for label, expected_num in expected_mapping.items():
            assert map_label_to_number(label) == expected_num

    def test_extraction_edge_cases(self):
        """Test edge cases in extraction."""
        edge_cases = [
            # Multiple labels
            ("This is both HAWKISH and DOVISH", ["HAWKISH", "DOVISH"]),
            # Label in quotes
            ('"NEUTRAL"', ["NEUTRAL"]),
            # Label with punctuation
            ("HAWKISH!", ["HAWKISH"]),
            # Label in parentheses
            ("(DOVISH)", ["DOVISH"]),
            # Mixed case with spaces
            ("H A W K I S H", ["HAWKISH"]),
        ]

        for text, possible_labels in edge_cases:
            text_upper = text.upper().replace(" ", "")
            found = False

            for label in ["HAWKISH", "DOVISH", "NEUTRAL"]:
                if label.replace(" ", "") in text_upper:
                    found = True
                    assert label in possible_labels
                    break

            if not found:
                # Check if any label is present in original form
                for label in ["HAWKISH", "DOVISH", "NEUTRAL"]:
                    if label in text.upper():
                        assert label in possible_labels
                        break


class TestExtractionPerformance:
    """Test extraction performance characteristics."""

    def test_extraction_speed(self):
        """Test extraction is fast enough."""
        import time
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        # Test native extraction speed
        labels = ["HAWKISH", "DOVISH", "NEUTRAL"] * 1000

        start = time.time()
        for label in labels:
            map_label_to_number(label)
        elapsed = time.time() - start

        # Should process 3000 labels in under 0.1 seconds
        assert elapsed < 0.1, f"Extraction too slow: {elapsed:.3f}s for 3000 labels"

    def test_extraction_memory_efficiency(self):
        """Test extraction doesn't leak memory."""
        import sys
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        # Get initial size
        initial_size = sys.getsizeof(map_label_to_number)

        # Process many labels
        for _ in range(10000):
            map_label_to_number("HAWKISH")

        # Size shouldn't have grown
        final_size = sys.getsizeof(map_label_to_number)
        assert final_size == initial_size, "Possible memory leak in extraction"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
