"""Unit tests for new output utilities and filename patterns."""

import re
from datetime import datetime

import pytest

from flame.config import TEST_OUTPUT_DIR
from flame.utils.output_utils import (
    build_output_filename,
    generate_output_path,
    parse_model_info,
    parse_output_filename,
)

pytestmark = pytest.mark.unit


def test_model_info_parsing():
    """Test model information parsing for different formats."""
    # Simple model
    provider, model_slug, model_family = parse_model_info("gpt-4")
    assert provider == "unknown"
    assert model_slug == "gpt-4"
    assert model_family is None

    # Provider/model
    provider, model_slug, model_family = parse_model_info("together_ai/Llama-2-7b")
    assert provider == "together_ai"
    assert model_slug == "llama-2-7b"
    assert model_family is None

    # Provider/family/model
    provider, model_slug, model_family = parse_model_info(
        "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )
    assert provider == "together_ai"
    assert model_slug == "llama-4-scout-17b-16e-instruct"
    assert model_family == "meta-llama"


def test_new_filename_pattern():
    """Test that new filenames follow the specified pattern."""
    filename = build_output_filename("llama-2-7b", "fomc", run=1, metrics=False)

    # Pattern: {model_slug}__{task_slug}__r{run:02d}__{yyyymmdd}__{uuid8}.csv
    pattern = r"^llama-2-7b__fomc__r01__\d{8}__[0-9a-f]{8}\.csv$"
    assert re.match(
        pattern, filename
    ), f"Filename {filename} doesn't match expected pattern"

    # Test metrics version
    metrics_filename = build_output_filename("llama-2-7b", "fomc", run=1, metrics=True)
    metrics_pattern = r"^llama-2-7b__fomc__r01__\d{8}__[0-9a-f]{8}_metrics\.csv$"
    assert re.match(metrics_pattern, metrics_filename)


def test_filename_parsing():
    """Test parsing filenames back to components."""
    filename = "llama-2-7b__fomc__r01__20250124__abcd1234.csv"
    parsed = parse_output_filename(filename)

    assert parsed["model_slug"] == "llama-2-7b"
    assert parsed["task_slug"] == "fomc"
    assert parsed["run"] == 1
    assert parsed["date"] == "20250124"
    assert parsed["uuid8"] == "abcd1234"
    assert parsed["metrics"] is False

    # Test metrics version
    metrics_filename = "llama-2-7b__fomc__r01__20250124__abcd1234_metrics.csv"
    metrics_parsed = parse_output_filename(metrics_filename)
    assert metrics_parsed["metrics"] is True


def test_new_directory_structure():
    """Test that new directory structure is created correctly."""
    # Test without model family
    path = generate_output_path(TEST_OUTPUT_DIR, "fomc", "together_ai/Llama-2-7b")
    expected_dir = TEST_OUTPUT_DIR / "fomc" / "together_ai"
    assert path.parent == expected_dir

    # Test with model family
    path = generate_output_path(
        TEST_OUTPUT_DIR, "fomc", "together_ai/meta-llama/Llama-4-Scout"
    )
    expected_dir = TEST_OUTPUT_DIR / "fomc" / "together_ai" / "meta-llama"
    assert path.parent == expected_dir


def test_round_trip_consistency():
    """Test that generation and parsing are consistent."""
    path = generate_output_path(
        TEST_OUTPUT_DIR,
        "causal_classification",
        "together_ai/meta-llama/Llama-2-7b",
        run=3,
        metrics=True,
    )

    parsed = parse_output_filename(path.name)
    assert parsed["task_slug"] == "causal_classification"
    assert parsed["model_slug"] == "llama-2-7b"
    assert parsed["run"] == 3
    assert parsed["metrics"] is True

    # Verify date is today
    today = datetime.now().strftime("%Y%m%d")
    assert parsed["date"] == today


def test_filename_uniqueness():
    """Test that generated filenames are unique due to UUID."""
    filename1 = build_output_filename("llama-2-7b", "fomc", run=1, metrics=False)
    filename2 = build_output_filename("llama-2-7b", "fomc", run=1, metrics=False)
    assert filename1 != filename2


def test_invalid_filename_parsing():
    """Test that invalid filenames raise appropriate errors."""
    with pytest.raises(ValueError, match="Invalid filename format"):
        parse_output_filename("invalid_filename.csv")

    with pytest.raises(ValueError, match="Invalid run format"):
        parse_output_filename("model__task__x01__20250124__abcd1234.csv")
