#!/usr/bin/env python3
"""Comprehensive test suite for sample_size functionality"""

import pytest
import tempfile
import yaml
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from datasets import Dataset

# Import system modules
from main import parse_arguments
from flame.code.fpb.fpb_inference import fpb_inference
from flame.code.fomc.fomc_inference import fomc_inference
from flame.code.inference import main as inference_main


class TestSampleSizeFeature:
    """Comprehensive tests for sample_size limiting functionality"""

    def test_cli_argument_parsing(self):
        """Test that sample_size argument is parsed correctly from CLI"""
        # Test with sample_size specified
        test_args = [
            "--mode",
            "inference",
            "--model",
            "test_model",
            "--sample_size",
            "10",
            "--tasks",
            "fpb",
            "fomc",
        ]
        with patch("sys.argv", ["main.py"] + test_args):
            args = parse_arguments()

        assert args.sample_size == 10
        assert args.tasks == ["fpb", "fomc"]

    def test_yaml_config_parsing(self):
        """Test that sample_size is loaded from YAML config"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            config = {
                "model": "test_model",
                "sample_size": 5,
                "tasks": ["fpb", "fomc"],
                "max_tokens": 100,
            }
            yaml.dump(config, f)
            f.flush()

            test_args = ["--config", f.name, "--mode", "inference"]
            with patch("sys.argv", ["main.py"] + test_args):
                args = parse_arguments()

            assert args.sample_size == 5
            assert args.tasks == ["fpb", "fomc"]
            assert args.max_tokens == 100

    def test_cli_overrides_yaml(self):
        """Test that CLI arguments override YAML config"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            config = {"model": "test_model", "sample_size": 20, "tasks": ["fpb"]}
            yaml.dump(config, f)
            f.flush()

            test_args = [
                "--config",
                f.name,
                "--mode",
                "inference",
                "--sample_size",
                "10",  # Override config
                "--tasks",
                "fpb",
                "fomc",  # Override config
            ]
            with patch("sys.argv", ["main.py"] + test_args):
                args = parse_arguments()

            assert args.sample_size == 10  # CLI override
            assert args.tasks == ["fpb", "fomc"]  # CLI override

    def test_no_sample_size_specified(self):
        """Test behavior when sample_size is not specified anywhere"""
        test_args = ["--mode", "inference", "--model", "test_model", "--tasks", "fpb"]
        with patch("sys.argv", ["main.py"] + test_args):
            args = parse_arguments()

        # Should default to None (use all samples)
        assert args.sample_size is None

    def test_invalid_sample_size(self):
        """Test handling of invalid sample_size values"""
        # Note: argparse doesn't validate negative integers by default
        # We might want to add custom validation in the future

        # Non-integer value
        test_args = [
            "--mode",
            "inference",
            "--model",
            "test_model",
            "--sample_size",
            "abc",
            "--tasks",
            "fpb",
        ]
        with patch("sys.argv", ["main.py"] + test_args):
            with pytest.raises(SystemExit):  # argparse should fail on non-int
                args = parse_arguments()

    @patch("litellm.completion")
    @patch("datasets.load_dataset")
    def test_fpb_inference_with_sample_size(self, mock_load_dataset, mock_completion):
        """Test FPB inference respects sample_size limit"""
        # Create mock dataset
        mock_data = []
        for i in range(100):
            mock_data.append(
                {
                    "sentence": f"Test sentence {i}",
                    "label": i % 3,  # 0, 1, 2 labels
                }
            )

        mock_dataset = {"test": Dataset.from_list(mock_data)}
        mock_load_dataset.return_value = mock_dataset

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "POSITIVE"
        mock_completion.return_value = [mock_response]

        # Test with sample_size=10
        args = Mock()
        args.sample_size = 10
        args.batch_size = 2
        args.prompt_format = "zero_shot"
        args.model = "test_model"

        result_df = fpb_inference(args)

        # Verify only 10 samples were processed
        assert len(result_df) == 10
        assert mock_completion.call_count == 5  # 10 samples / batch_size 2

    @patch("litellm.completion")
    @patch("datasets.load_dataset")
    def test_fpb_inference_without_sample_size(
        self, mock_load_dataset, mock_completion
    ):
        """Test FPB inference uses all data when sample_size is None"""
        # Create mock dataset
        mock_data = []
        for i in range(50):
            mock_data.append({"sentence": f"Test sentence {i}", "label": i % 3})

        mock_dataset = {"test": Dataset.from_list(mock_data)}
        mock_load_dataset.return_value = mock_dataset

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "POSITIVE"
        mock_completion.return_value = [mock_response]

        # Test without sample_size
        args = Mock()
        args.sample_size = None  # Use all data
        args.batch_size = 5
        args.prompt_format = "zero_shot"
        args.model = "test_model"

        result_df = fpb_inference(args)

        # Verify all 50 samples were processed
        assert len(result_df) == 50
        assert mock_completion.call_count == 10  # 50 samples / batch_size 5

    @patch("litellm.completion")
    @patch("datasets.load_dataset")
    def test_sample_size_exceeds_dataset(self, mock_load_dataset, mock_completion):
        """Test behavior when sample_size exceeds dataset size"""
        # Create mock dataset with only 5 samples
        mock_data = []
        for i in range(5):
            mock_data.append({"sentence": f"Test sentence {i}", "label": i % 3})

        mock_dataset = {"test": Dataset.from_list(mock_data)}
        mock_load_dataset.return_value = mock_dataset

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "POSITIVE"
        mock_completion.return_value = [mock_response]

        # Request 100 samples but dataset only has 5
        args = Mock()
        args.sample_size = 100
        args.batch_size = 1
        args.prompt_format = "zero_shot"
        args.model = "test_model"

        result_df = fpb_inference(args)

        # Should only process the 5 available samples
        assert len(result_df) == 5
        assert mock_completion.call_count == 5

    @patch("litellm.completion")
    @patch("datasets.load_dataset")
    def test_fomc_inference_with_sample_size(self, mock_load_dataset, mock_completion):
        """Test FOMC inference respects sample_size limit"""
        # Create mock dataset
        mock_data = []
        for i in range(50):
            mock_data.append(
                {
                    "sentence": f"FOMC statement {i}",
                    "label": ["DOVISH", "HAWKISH", "NEUTRAL"][i % 3],
                }
            )

        mock_dataset = {"test": Dataset.from_list(mock_data)}
        mock_load_dataset.return_value = mock_dataset

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "HAWKISH"
        mock_completion.return_value = [mock_response]

        # Test with sample_size=15
        args = Mock()
        args.sample_size = 15
        args.batch_size = 3
        args.prompt_format = "zero_shot"
        args.model = "test_model"

        result_df = fomc_inference(args)

        # Verify only 15 samples were processed
        assert len(result_df) == 15
        assert mock_completion.call_count == 5  # 15 samples / batch_size 3

    @patch("litellm.completion")
    @patch("datasets.load_dataset")
    def test_zero_sample_size(self, mock_load_dataset, mock_completion):
        """Test edge case with sample_size=0"""
        # Create mock dataset
        mock_data = []
        for i in range(10):
            mock_data.append({"sentence": f"Test sentence {i}", "label": i % 3})

        mock_dataset = {"test": Dataset.from_list(mock_data)}
        mock_load_dataset.return_value = mock_dataset

        # Test with sample_size=0
        args = Mock()
        args.sample_size = 0
        args.batch_size = 1
        args.prompt_format = "zero_shot"
        args.model = "test_model"

        result_df = fpb_inference(args)

        # Should process 0 samples
        assert len(result_df) == 0
        assert mock_completion.call_count == 0

    @patch("flame.code.fpb.fpb_inference.fpb_inference")
    @patch("flame.code.fomc.fomc_inference.fomc_inference")
    def test_multi_task_with_sample_size(self, mock_fomc, mock_fpb):
        """Test running multiple tasks with sample_size"""
        # Mock the inference functions to return dummy DataFrames
        mock_fpb.return_value = pd.DataFrame(
            {
                "sentences": ["test"] * 10,
                "llm_responses": ["POSITIVE"] * 10,
                "actual_labels": [0] * 10,
                "complete_responses": [None] * 10,
            }
        )
        mock_fomc.return_value = pd.DataFrame(
            {
                "sentences": ["test"] * 10,
                "llm_responses": ["HAWKISH"] * 10,
                "actual_labels": [1] * 10,
                "complete_responses": [None] * 10,
            }
        )

        # Create args
        args = Mock()
        args.sample_size = 10
        args.model = "test_model"
        args.tasks = ["fpb", "fomc"]
        args.batch_size = 2
        args.max_tokens = 100
        args.temperature = 0.0
        args.top_p = 0.9
        args.top_k = None
        args.repetition_penalty = 1.0
        args.prompt_format = "zero_shot"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("flame.config.RESULTS_DIR", Path(tmpdir)):
                with patch("flame.config.TEST_OUTPUT_DIR", Path(tmpdir)):
                    with patch("flame.config.IN_PYTEST", False):
                        # Run fpb task
                        args.task = "fpb"
                        inference_main(args)

                        # Run fomc task
                        args.task = "fomc"
                        inference_main(args)

                        # Verify both tasks were called with correct args
                        assert mock_fpb.called
                        assert mock_fomc.called

                        # Check args passed to inference functions
                        fpb_args = mock_fpb.call_args[0][0]
                        fomc_args = mock_fomc.call_args[0][0]

                        assert hasattr(fpb_args, "sample_size")
                        assert fpb_args.sample_size == 10
                        assert hasattr(fomc_args, "sample_size")
                        assert fomc_args.sample_size == 10

    def test_dataset_select_method(self):
        """Test that HuggingFace Dataset.select() works as expected"""
        # Create a test dataset
        data = [{"text": f"sample {i}", "label": i} for i in range(20)]
        dataset = Dataset.from_list(data)

        # Test selecting first 5 samples
        subset = dataset.select(range(min(5, len(dataset))))
        assert len(subset) == 5
        assert subset[0]["text"] == "sample 0"
        assert subset[4]["text"] == "sample 4"

        # Test selecting more than available
        subset = dataset.select(range(min(100, len(dataset))))
        assert len(subset) == 20  # Full dataset

    @patch("litellm.completion")
    @patch("datasets.load_dataset")
    def test_inference_output_with_sample_size(
        self, mock_load_dataset, mock_completion
    ):
        """Test that inference output CSV contains correct number of samples"""
        # Create mock dataset
        mock_data = []
        for i in range(30):
            mock_data.append({"sentence": f"Test sentence {i}", "label": i % 3})

        mock_dataset = {"test": Dataset.from_list(mock_data)}
        mock_load_dataset.return_value = mock_dataset

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "POSITIVE"
        mock_completion.return_value = [mock_response]

        # Test with sample_size=10
        args = Mock()
        args.sample_size = 10
        args.batch_size = 2
        args.prompt_format = "zero_shot"
        args.model = "test_model"
        args.task = "fpb"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("flame.config.RESULTS_DIR", Path(tmpdir)):
                with patch("flame.config.TEST_OUTPUT_DIR", Path(tmpdir)):
                    with patch("flame.config.IN_PYTEST", False):
                        # Run inference
                        inference_main(args)

                        # Check output file
                        output_files = list(Path(tmpdir).glob("fpb/*.csv"))
                        assert len(output_files) == 1

                        # Read the CSV and verify sample count
                        df = pd.read_csv(output_files[0])
                        assert len(df) == 10


if __name__ == "__main__":
    # Run a simple test to verify the implementation
    test = TestSampleSizeFeature()
    test.test_cli_argument_parsing()
    print("Basic test passed!")
