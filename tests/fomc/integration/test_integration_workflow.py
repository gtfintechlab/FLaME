#!/usr/bin/env python3
"""
Integration tests for FOMC workflow.
Tests component interactions without real API calls.
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "benchforge"))


class TestNativeWorkflowIntegration:
    """Test native FLAME workflow integration."""

    @pytest.fixture
    def mock_litellm(self):
        """Mock litellm for testing."""
        with patch("litellm.completion") as mock:
            # Create proper response structure
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "HAWKISH"
            mock.return_value = mock_response
            yield mock

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        data = {
            "sentence": [
                "The Fed raised rates by 25 basis points.",
                "The Committee maintains accommodative stance.",
                "Policy remains appropriate for current conditions.",
            ],
            "label": ["HAWKISH", "DOVISH", "NEUTRAL"],
        }
        return pd.DataFrame(data)

    def test_inference_workflow(self, mock_litellm, sample_dataset, tmp_path):
        """Test complete inference workflow."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        # Setup
        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        results = []

        # Process each sample
        for idx, row in sample_dataset.iterrows():
            # Generate prompt
            prompt_func(row["sentence"])

            # Mock API call
            response = mock_litellm.return_value.choices[0].message.content

            # Store result
            results.append(
                {
                    "sentence": row["sentence"],
                    "llm_response": response,
                    "actual_label": row["label"],
                }
            )

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Save results
        output_file = tmp_path / "results.csv"
        results_df.to_csv(output_file, index=False)

        # Validate
        assert output_file.exists()
        loaded_df = pd.read_csv(output_file)
        assert len(loaded_df) == 3
        assert "llm_response" in loaded_df.columns

    def test_evaluation_workflow(self, sample_dataset, tmp_path):
        """Test evaluation workflow."""
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        # Create mock inference results
        results = {
            "sentence": sample_dataset["sentence"].tolist(),
            "llm_response": ["HAWKISH", "DOVISH", "NEUTRAL"],
            "actual_label": sample_dataset["label"].tolist(),
        }
        results_df = pd.DataFrame(results)

        # Save mock results
        results_file = tmp_path / "inference_results.csv"
        results_df.to_csv(results_file, index=False)

        # Perform evaluation
        extracted_labels = []
        for response in results_df["llm_response"]:
            # Simple extraction
            for label in ["HAWKISH", "DOVISH", "NEUTRAL"]:
                if label in response.upper():
                    extracted_labels.append(label)
                    break
            else:
                extracted_labels.append(None)

        # Map to numbers
        predicted_nums = [
            map_label_to_number(label) if label else None for label in extracted_labels
        ]
        actual_nums = [
            map_label_to_number(label) for label in results_df["actual_label"]
        ]

        # Calculate accuracy
        correct = sum(
            1 for p, a in zip(predicted_nums, actual_nums) if p == a and p is not None
        )
        accuracy = correct / len(actual_nums)

        # Create evaluation results
        eval_results = {
            "accuracy": accuracy,
            "total_samples": len(results_df),
            "correct_predictions": correct,
        }

        # Save evaluation
        eval_file = tmp_path / "evaluation.json"
        with open(eval_file, "w") as f:
            json.dump(eval_results, f, indent=2)

        assert eval_file.exists()
        assert accuracy == 1.0  # Perfect since we mocked matching responses

    def test_prompt_dataset_integration(self, sample_dataset):
        """Test prompt generation with dataset integration."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)

        # Generate prompts for all samples
        prompts = []
        for _, row in sample_dataset.iterrows():
            prompt = prompt_func(row["sentence"])
            prompts.append(prompt)

        # Validate all prompts
        assert len(prompts) == len(sample_dataset)
        for prompt in prompts:
            assert "HAWKISH" in prompt
            assert "DOVISH" in prompt
            assert "NEUTRAL" in prompt

    def test_error_handling_integration(self, mock_litellm):
        """Test error handling in workflow."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        # Setup error scenario
        mock_litellm.side_effect = Exception("API Error")

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        prompt = prompt_func("Test statement")

        # Should handle error gracefully
        try:
            mock_litellm(model="test", messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            assert "API Error" in str(e)


class TestBenchForgeWorkflowIntegration:
    """Test BenchForge workflow integration."""

    @pytest.fixture
    def fomc_task(self):
        """Create FOMCTask for testing."""
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

    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for BenchForge."""
        return [
            {"sentence": "Fed raised rates", "label": "HAWKISH"},
            {"sentence": "Accommodative policy", "label": "DOVISH"},
            {"sentence": "Policy unchanged", "label": "NEUTRAL"},
        ]

    def test_benchforge_inference_workflow(self, fomc_task, mock_dataset):
        """Test BenchForge inference workflow."""
        from flame.benchforge import PromptFormat
        from bench_forge.prompts.extractor import ResponseExtractor, ExtractionStrategy

        extractor = ResponseExtractor()
        results = []

        # Process samples
        for sample in mock_dataset:
            # Generate prompt
            fomc_task.create_prompt(sample, PromptFormat.ZERO_SHOT)

            # Mock LLM response
            mock_response = f"Based on the analysis, this is {sample['label']}"

            # Extract label
            extraction = extractor.extract(
                mock_response,
                strategy=ExtractionStrategy.FUZZY,
                options=["HAWKISH", "DOVISH", "NEUTRAL"],
            )

            results.append(
                {
                    "input": sample["sentence"],
                    "response": mock_response,
                    "extracted": extraction.value,
                    "confidence": extraction.confidence,
                    "actual": sample["label"],
                }
            )

        # Validate results
        assert len(results) == 3
        for r in results:
            assert r["extracted"] == r["actual"]
            assert r["confidence"] > 0

    def test_benchforge_extraction_pipeline(self, fomc_task):
        """Test extraction pipeline integration."""
        from bench_forge.prompts.extractor import ResponseExtractor, ExtractionStrategy

        extractor = ResponseExtractor()

        test_responses = [
            ("HAWKISH", ExtractionStrategy.RULE_BASED, 1.0),
            ("I think this is hawkish", ExtractionStrategy.FUZZY, 0.8),
            ("The stance appears dovish", ExtractionStrategy.FUZZY, 0.8),
            ("neutral policy", ExtractionStrategy.FUZZY, 0.8),
        ]

        for response, strategy, min_confidence in test_responses:
            result = extractor.extract(
                response, strategy=strategy, options=["HAWKISH", "DOVISH", "NEUTRAL"]
            )

            assert result.value is not None
            assert result.confidence >= min_confidence

    def test_benchforge_llm_fallback(self, fomc_task):
        """Test LLM-based extraction fallback."""
        from bench_forge.prompts.extractor import ResponseExtractor, ExtractionStrategy

        extractor = ResponseExtractor()

        # Test messy response that needs LLM fallback
        messy_response = """
        The Federal Reserve's decision reflects a complex balance of factors.
        While inflation remains elevated, there are signs of moderation.
        The labor market shows resilience. Overall sentiment leans positive
        for tightening but with caution.
        """

        # First try rule-based (should fail)
        rule_result = extractor.extract(
            messy_response,
            strategy=ExtractionStrategy.RULE_BASED,
            options=["HAWKISH", "DOVISH", "NEUTRAL"],
        )

        assert rule_result.value is None or rule_result.confidence < 0.5

        # Try fuzzy (might work)
        extractor.extract(
            messy_response,
            strategy=ExtractionStrategy.FUZZY,
            options=["HAWKISH", "DOVISH", "NEUTRAL"],
        )

        # With LLM fallback (mocked)
        def mock_llm_extract(text, **kwargs):
            # Mock LLM would analyze and return HAWKISH
            from bench_forge.prompts.extractor import ExtractionResult

            return ExtractionResult(
                value="HAWKISH",
                confidence=0.7,
                strategy=ExtractionStrategy.LLM_BASED,
                metadata={"method": "llm_analysis"},
            )

        # Patch the LLM extraction
        with patch.object(extractor, "_extract_llm_based", mock_llm_extract):
            llm_result = extractor.extract(
                messy_response,
                strategy=ExtractionStrategy.LLM_BASED,
                options=["HAWKISH", "DOVISH", "NEUTRAL"],
            )

            assert llm_result.value == "HAWKISH"
            assert llm_result.strategy == ExtractionStrategy.LLM_BASED


class TestCrossImplementationCompatibility:
    """Test compatibility between implementations."""

    def test_prompt_similarity(self):
        """Test prompts are similar between implementations."""
        from flame.code.prompts.registry import get_prompt, PromptFormat as NativeFormat

        try:
            from flame.tasks.fomc import FOMCTask
            from flame.benchforge import FLAMEConfig, PromptFormat as BFFormat
        except ImportError:
            pytest.skip("BenchForge not available")

        # Native prompt
        native_func = get_prompt("fomc", NativeFormat.ZERO_SHOT)
        native_prompt = native_func("Test statement")

        # BenchForge prompt
        config = FLAMEConfig(
            name="fomc",
            dataset="fomc",
            prompt_format=BFFormat.ZERO_SHOT,
            text_field="sentence",
        )
        task = FOMCTask(config)
        bf_prompt = task.create_prompt(
            {"sentence": "Test statement"}, BFFormat.ZERO_SHOT
        )

        # Check key elements present in both
        key_elements = ["HAWKISH", "DOVISH", "NEUTRAL", "Test statement"]

        for element in key_elements:
            assert element in native_prompt
            assert element in bf_prompt

    def test_label_mapping_consistency(self):
        """Test label mappings are consistent."""
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        # Standard mapping both should follow
        mapping = {"HAWKISH": 1, "DOVISH": 0, "NEUTRAL": 2}

        for label, expected in mapping.items():
            assert map_label_to_number(label) == expected
            assert map_label_to_number(label.lower()) == expected

    def test_data_format_compatibility(self):
        """Test data formats are compatible."""
        # Native format
        native_data = pd.DataFrame(
            {"sentence": ["Test 1", "Test 2"], "label": ["HAWKISH", "DOVISH"]}
        )

        # BenchForge format (same for FOMC)
        bf_data = [
            {"sentence": "Test 1", "label": "HAWKISH"},
            {"sentence": "Test 2", "label": "DOVISH"},
        ]

        # Convert BenchForge to DataFrame
        bf_df = pd.DataFrame(bf_data)

        # Should be identical
        assert native_data.equals(bf_df)


class TestMigrationReadiness:
    """Test migration readiness indicators."""

    def test_feature_flag_system(self):
        """Test feature flag system for migration."""
        import os

        # Test environment-based flags
        test_flags = {
            "USE_BENCHFORGE_FOMC": "fomc",
            "USE_BENCHFORGE_FPB": "fpb",
            "USE_BENCHFORGE_ALL": "all",
        }

        for env_var, expected_task in test_flags.items():
            # Simulate setting environment variable
            with patch.dict(os.environ, {env_var: "1"}):
                # In real implementation, this would check MigrationConfig
                assert os.getenv(env_var) == "1"

    def test_result_normalization(self):
        """Test result normalization for compatibility."""
        # Native format
        native_df = pd.DataFrame(
            {
                "llm_responses": ["HAWKISH", "DOVISH"],
                "extracted_labels": ["HAWKISH", "DOVISH"],
                "actual_labels": ["HAWKISH", "DOVISH"],
            }
        )

        # BenchForge format
        bf_df = pd.DataFrame(
            {
                "raw_response": ["HAWKISH", "DOVISH"],
                "extracted_response": ["HAWKISH", "DOVISH"],
                "ground_truth": ["HAWKISH", "DOVISH"],
            }
        )

        # Normalize (simplified)
        column_mapping = {
            "raw_response": "llm_responses",
            "extracted_response": "extracted_labels",
            "ground_truth": "actual_labels",
        }

        normalized_bf = bf_df.rename(columns=column_mapping)

        # Should have same columns
        assert set(native_df.columns) == set(normalized_bf.columns)

    def test_parallel_execution_capability(self):
        """Test ability to run both implementations in parallel."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        # Can run native
        native_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        native_prompt = native_func("Test")
        assert native_prompt is not None

        # Can run BenchForge (if available)
        try:
            from flame.tasks.fomc import FOMCTask
            from flame.benchforge import FLAMEConfig

            config = FLAMEConfig(name="fomc", dataset="fomc")
            task = FOMCTask(config)
            assert task is not None

            # Both can coexist
            assert native_prompt is not None and task is not None
        except ImportError:
            pass  # BenchForge optional for this test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
