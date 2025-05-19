#!/usr/bin/env python3
"""Test backward compatibility - ensure system works without sample_size"""

from unittest.mock import Mock, patch
from datasets import Dataset
import pandas as pd

from flame.code.fpb.fpb_inference import fpb_inference
from flame.code.fomc.fomc_inference import fomc_inference


class TestBackwardCompatibility:
    """Ensure the system still works when sample_size is not specified"""

    @patch("litellm.completion")
    @patch("datasets.load_dataset")
    def test_fpb_inference_without_sample_size_attribute(
        self, mock_load_dataset, mock_completion
    ):
        """Test FPB inference when args doesn't have sample_size attribute"""
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

        # Args without sample_size attribute (old version)
        args = Mock(spec=["batch_size", "prompt_format", "model"])
        args.batch_size = 5
        args.prompt_format = "zero_shot"
        args.model = "test_model"

        # Should not raise AttributeError
        result_df = fpb_inference(args)

        # Should process all 50 samples
        assert len(result_df) == 50

    @patch("litellm.completion")
    @patch("datasets.load_dataset")
    def test_fomc_inference_backward_compatibility(
        self, mock_load_dataset, mock_completion
    ):
        """Test FOMC inference maintains backward compatibility"""
        # Create mock dataset
        mock_data = []
        for i in range(30):
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

        # Old-style args without sample_size
        args = Mock()
        args.batch_size = 3
        args.prompt_format = "zero_shot"
        args.model = "test_model"
        # Explicitly don't set sample_size

        # Should work without errors
        result_df = fomc_inference(args)

        # Should process all 30 samples
        assert len(result_df) == 30

    def test_main_parse_arguments_backward_compatible(self):
        """Test that parse_arguments still works without sample_size"""
        from main import parse_arguments

        # Test old-style command line args
        test_args = [
            "--mode",
            "inference",
            "--model",
            "test_model",
            "--tasks",
            "fpb",
            # No --sample_size
        ]

        with patch("sys.argv", ["main.py"] + test_args):
            args = parse_arguments()

        # Should have default value (None)
        assert args.sample_size is None

        # Other args should work
        assert args.mode == "inference"
        assert args.model == "test_model"
        assert args.tasks == ["fpb"]

    def test_config_without_sample_size(self):
        """Test loading config file without sample_size field"""
        import tempfile
        import yaml
        from main import parse_arguments

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            # Old-style config without sample_size
            config = {
                "model": "test_model",
                "tasks": ["fpb", "fomc"],
                "max_tokens": 128,
                "temperature": 0.0,
            }
            yaml.dump(config, f)
            f.flush()

            test_args = ["--config", f.name, "--mode", "inference"]

            with patch("sys.argv", ["main.py"] + test_args):
                args = parse_arguments()

            # Should have default None value
            assert args.sample_size is None

            # Other config values should load
            assert args.model == "test_model"
            assert args.tasks == ["fpb", "fomc"]
            assert args.max_tokens == 128

    def test_mixed_args_old_and_new(self):
        """Test mixing old args with new sample_size"""
        from main import parse_arguments
        import tempfile
        import yaml

        # Old config, new CLI arg
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            config = {
                "model": "test_model",
                "tasks": ["fpb"],
                "batch_size": 10,
                # No sample_size in config
            }
            yaml.dump(config, f)
            f.flush()

            test_args = [
                "--config",
                f.name,
                "--mode",
                "inference",
                "--sample_size",
                "5",  # New CLI arg
            ]

            with patch("sys.argv", ["main.py"] + test_args):
                args = parse_arguments()

            # Should use CLI value
            assert args.sample_size == 5

            # Old config values should still work
            assert args.batch_size == 10

    @patch("flame.code.fpb.fpb_inference.fpb_inference")
    def test_run_tasks_backward_compatible(self, mock_fpb):
        """Test run_tasks function works with old-style args"""
        from main import run_tasks

        # Mock inference to return dummy data
        mock_fpb.return_value = pd.DataFrame(
            {"sentences": ["test"] * 10, "llm_responses": ["POSITIVE"] * 10}
        )

        # Old-style args
        args = Mock()
        args.model = "test_model"
        args.batch_size = 5
        args.max_tokens = 100
        args.temperature = 0.0
        args.top_p = 0.9
        args.top_k = None
        args.repetition_penalty = 1.0
        args.prompt_format = "zero_shot"
        # No sample_size attribute

        # Should work without errors
        run_tasks(["fpb"], "inference", args)

        # Verify inference was called
        assert mock_fpb.called

        # Check that args were passed through
        call_args = mock_fpb.call_args[0][0]
        assert hasattr(call_args, "model")
        assert call_args.model == "test_model"


if __name__ == "__main__":
    # Run a test
    test = TestBackwardCompatibility()
    print("Backward compatibility tests created!")
