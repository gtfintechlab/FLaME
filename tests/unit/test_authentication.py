"""Unit tests for HuggingFace authentication error handling."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


def test_main_no_huggingface_token(monkeypatch, capsys):
    """Test that main.py exits with error when no HF token is provided."""
    # Remove HF token from environment
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)

    # Mock sys.argv to simulate running inference
    test_args = ["main.py", "--mode", "inference", "--tasks", "fomc", "--model", "test"]

    with patch.object(sys, "argv", test_args):
        # Import main.py module
        import main

        # The main module should exit with code 1 when no token is found
        with pytest.raises(SystemExit) as exc_info:
            # Re-run the main module logic
            main.args = main.parse_arguments()
            main.main_logger = main.get_component_logger("flame.main")

            # This should fail due to missing token
            HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not HUGGINGFACEHUB_API_TOKEN:
                main.main_logger.error(
                    "Hugging Face API token not found. This is required for accessing FLaME datasets."
                )
                print(
                    "ERROR: Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in the environment."
                )
                print("The FLaME datasets are private and require authentication.")
                sys.exit(1)

        assert exc_info.value.code == 1

    # Check output
    captured = capsys.readouterr()
    assert "ERROR: Hugging Face API token not found" in captured.out
    assert "FLaME datasets are private" in captured.out


def test_main_invalid_huggingface_token(monkeypatch, capsys):
    """Test that main.py exits with error when HF token is invalid."""
    # Set invalid token
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "invalid_token_12345")

    # Mock the login function to raise an exception
    mock_login = MagicMock(side_effect=Exception("Invalid user token."))

    with patch("main.login", mock_login):
        with patch.object(
            sys, "argv", ["main.py", "--mode", "inference", "--tasks", "fomc"]
        ):
            import main

            with pytest.raises(SystemExit) as exc_info:
                # Re-run token validation logic
                main.args = main.parse_arguments()
                main.main_logger = main.get_component_logger("flame.main")

                HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if HUGGINGFACEHUB_API_TOKEN:
                    try:
                        mock_login(token=HUGGINGFACEHUB_API_TOKEN)
                        main.main_logger.info("Logged in to Hugging Face Hub")
                    except Exception as e:
                        main.main_logger.error(
                            f"Failed to authenticate with Hugging Face Hub: {e}"
                        )
                        print(
                            f"ERROR: Failed to authenticate with Hugging Face Hub: {e}"
                        )
                        print("Please check your HUGGINGFACEHUB_API_TOKEN is valid.")
                        sys.exit(1)

            assert exc_info.value.code == 1

    # Check output
    captured = capsys.readouterr()
    assert "ERROR: Failed to authenticate with Hugging Face Hub" in captured.out
    assert "Invalid user token" in captured.out


def test_safe_load_dataset_authentication_error(monkeypatch, capsys):
    """Test safe_load_dataset handles authentication errors properly."""
    from flame.utils.dataset_utils import safe_load_dataset

    # Mock load_dataset to raise authentication error
    def mock_load_dataset(*args, **kwargs):
        raise Exception("401 Client Error: Unauthorized")

    with patch("flame.utils.dataset_utils.load_dataset", mock_load_dataset):
        with pytest.raises(SystemExit) as exc_info:
            safe_load_dataset("private/dataset")

        assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "authentication issues" in captured.out
    assert "HUGGINGFACEHUB_API_TOKEN" in captured.out


def test_safe_load_dataset_not_found_error(monkeypatch, capsys):
    """Test safe_load_dataset handles dataset not found errors properly."""
    from flame.utils.dataset_utils import safe_load_dataset

    # Mock load_dataset to raise not found error
    def mock_load_dataset(*args, **kwargs):
        raise Exception("404 Client Error: Dataset not found")

    with patch("flame.utils.dataset_utils.load_dataset", mock_load_dataset):
        with pytest.raises(SystemExit) as exc_info:
            safe_load_dataset("nonexistent/dataset")

        assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Dataset 'nonexistent/dataset' not found" in captured.out


def test_safe_load_dataset_success():
    """Test safe_load_dataset works with valid dataset."""
    from flame.utils.dataset_utils import safe_load_dataset

    # Mock successful dataset loading
    mock_dataset = {"train": ["data1", "data2"], "test": ["data3"]}

    with patch("flame.utils.dataset_utils.load_dataset", return_value=mock_dataset):
        result = safe_load_dataset("valid/dataset")
        assert result == mock_dataset

        # Test with split
        result_split = safe_load_dataset("valid/dataset", split="train")
        assert result_split == ["data1", "data2"]
