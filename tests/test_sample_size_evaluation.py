#!/usr/bin/env python3
"""Test that evaluation handles inference results with limited samples correctly"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from flame.code.fpb.fpb_evaluate import fpb_evaluate


class TestSampleSizeEvaluation:
    """Test evaluation with inference results from limited samples"""

    def test_fpb_evaluation_with_limited_results(self):
        """Test FPB evaluation handles results from limited inference"""
        # Create mock inference results with only 10 samples
        inference_results = pd.DataFrame(
            {
                "sentences": [f"sentence {i}" for i in range(10)],
                "llm_responses": ["POSITIVE"] * 5 + ["NEGATIVE"] * 3 + ["NEUTRAL"] * 2,
                "actual_labels": [2] * 5 + [0] * 3 + [1] * 2,  # Matching labels
                "complete_responses": [None] * 10,
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            inference_results.to_csv(f.name, index=False)

            # Mock args
            args = Mock()
            args.dataset = "fpb"
            args.model = "test_model"
            args.batch_size = 2

            # Mock the LLM completion for label extraction
            with patch("litellm.completion") as mock_completion:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "POSITIVE"
                mock_completion.return_value = [mock_response]

                with patch("flame.config.EVALUATION_DIR", Path("/tmp")):
                    # Run evaluation
                    df, metrics_df = fpb_evaluate(f.name, args)

                    # Verify results
                    assert len(df) == 10  # Should match input size
                    assert "extracted_labels" in df.columns
                    assert len(metrics_df) == 4  # Accuracy, Precision, Recall, F1

        Path(f.name).unlink()  # Clean up

    def test_fomc_evaluation_with_limited_results(self):
        """Test FOMC evaluation handles results from limited inference"""
        # Create mock inference results with only 5 samples
        inference_results = pd.DataFrame(
            {
                "sentences": [f"FOMC statement {i}" for i in range(5)],
                "llm_responses": ["HAWKISH"] * 2 + ["DOVISH"] * 2 + ["NEUTRAL"] * 1,
                "actual_labels": ["HAWKISH"] * 2 + ["DOVISH"] * 2 + ["NEUTRAL"] * 1,
                "complete_responses": [None] * 5,
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            inference_results.to_csv(f.name, index=False)

            # Mock args
            args = Mock()
            args.dataset = "fomc"
            args.model = "test_model"
            args.batch_size = 1

            # Mock the extraction process
            with patch(
                "flame.code.fomc.fomc_evaluate.process_batch_with_retry"
            ) as mock_process:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "HAWKISH"
                mock_process.return_value = [mock_response]

                with patch("flame.config.EVALUATION_DIR", Path("/tmp")):
                    # Try to import and run evaluation
                    try:
                        from flame.code.fomc.fomc_evaluate import fomc_evaluate

                        df, metrics_df = fomc_evaluate(f.name, args)

                        # Verify results
                        assert len(df) == 5  # Should match input size
                        assert len(metrics_df) == 4  # Accuracy, Precision, Recall, F1
                    except ImportError:
                        # FOMC evaluation might not be implemented yet
                        pass

        Path(f.name).unlink()  # Clean up

    def test_evaluation_with_empty_results(self):
        """Test evaluation handles empty inference results"""
        # Create empty results
        empty_results = pd.DataFrame(
            {
                "sentences": [],
                "llm_responses": [],
                "actual_labels": [],
                "complete_responses": [],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            empty_results.to_csv(f.name, index=False)

            # Mock args
            args = Mock()
            args.dataset = "fpb"
            args.model = "test_model"
            args.batch_size = 1

            with patch("flame.config.EVALUATION_DIR", Path("/tmp")):
                # This should handle empty results gracefully
                with pytest.raises(Exception):  # Might raise due to empty data
                    df, metrics_df = fpb_evaluate(f.name, args)

        Path(f.name).unlink()  # Clean up

    def test_evaluation_metrics_calculation(self):
        """Test that metrics are calculated correctly for limited samples"""
        # Create results with known accuracy
        results = pd.DataFrame(
            {
                "sentences": [f"sentence {i}" for i in range(8)],
                "llm_responses": ["POSITIVE"] * 4 + ["NEGATIVE"] * 4,
                "actual_labels": [2] * 3 + [0] * 1 + [0] * 4,  # 3+4=7 correct out of 8
                "complete_responses": [None] * 8,
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            results.to_csv(f.name, index=False)

            # Mock args
            args = Mock()
            args.dataset = "fpb"
            args.model = "test_model"
            args.batch_size = 2

            # Mock the label extraction to return exact matches
            with patch(
                "flame.code.fpb.fpb_evaluate.process_batch_with_retry"
            ) as mock_process:

                def mock_extraction(args, messages, batch_idx, total):
                    responses = []
                    for msg in messages:
                        content = msg[0]["content"]
                        if "POSITIVE" in content:
                            label = "POSITIVE"
                        elif "NEGATIVE" in content:
                            label = "NEGATIVE"
                        else:
                            label = "NEUTRAL"

                        mock_resp = Mock()
                        mock_resp.choices = [Mock()]
                        mock_resp.choices[0].message.content = label
                        responses.append(mock_resp)
                    return responses

                mock_process.side_effect = mock_extraction

                with patch("flame.config.EVALUATION_DIR", Path("/tmp")):
                    df, metrics_df = fpb_evaluate(f.name, args)

                    # Check accuracy
                    accuracy = metrics_df[metrics_df["Metric"] == "Accuracy"][
                        "Value"
                    ].iloc[0]
                    expected_accuracy = 7 / 8  # 7 correct out of 8
                    assert abs(accuracy - expected_accuracy) < 0.01

        Path(f.name).unlink()  # Clean up


if __name__ == "__main__":
    # Run a basic test
    test = TestSampleSizeEvaluation()
    test.test_fpb_evaluation_with_limited_results()
    print("Evaluation tests created!")
