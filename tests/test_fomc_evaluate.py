"""Tests for FOMC evaluation functionality."""

import pytest
import pandas as pd
from ferrari.code.fomc.fomc_evaluate import (
    map_label_to_number,
    extraction_prompt,
    validate_input_data,
    fomc_evaluate,
)
from tests.utils.fixtures import (
    create_mock_completion,
    setup_test_data,
    assert_metrics_in_range,
)
from tests.utils.constants import FOMC_TEST_CASES


# Basic function tests
def test_map_label_to_number():
    """Test label mapping function."""
    assert map_label_to_number("DOVISH") == 0
    assert map_label_to_number("HAWKISH") == 1
    assert map_label_to_number("NEUTRAL") == 2
    assert map_label_to_number("invalid") == -1
    assert map_label_to_number("") == -1


def test_extraction_prompt():
    """Test prompt generation."""
    response = "The Fed's stance appears dovish"
    prompt = extraction_prompt(response)
    assert "HAWKISH" in prompt
    assert "DOVISH" in prompt
    assert "NEUTRAL" in prompt
    assert response in prompt


def test_validate_input_data():
    """Test data validation."""
    valid_df = pd.DataFrame({"llm_responses": ["response1"], "actual_labels": [0]})
    validate_input_data(valid_df)  # Should not raise

    with pytest.raises(ValueError):
        invalid_df = pd.DataFrame({"llm_responses": ["response1"]})
        validate_input_data(invalid_df)


# Main evaluation tests
@pytest.mark.task_args({"dataset": "fomc"})
@pytest.mark.parametrize("test_case", FOMC_TEST_CASES)
def test_evaluation(tmp_path, task_args, monkeypatch, test_case):
    """Test the evaluation pipeline with minimal examples."""
    # Setup test data with pre-generated responses
    test_data = {
        "llm_responses": test_case["llm_responses"],
        "actual_labels": test_case["actual_labels"],
    }
    test_file = setup_test_data(tmp_path, test_data)

    # Mock completion to return expected labels
    mock_func = create_mock_completion(test_case["mock_responses"])
    monkeypatch.setattr(
        "ferrari.together_code.fomc.fomc_evaluate.completion", mock_func
    )

    # Run evaluation
    try:
        results_df, metrics_df = fomc_evaluate(str(test_file), task_args)

        # Verify results
        assert len(results_df) == len(test_case["llm_responses"])
        assert "extracted_labels" in results_df.columns
        assert_metrics_in_range(metrics_df, "Accuracy", test_case["expected_accuracy"])

    except Exception as e:
        pytest.fail(f"Evaluation failed: {str(e)}")


@pytest.mark.task_args({"dataset": "fomc"})
def test_invalid_responses(tmp_path, task_args, monkeypatch):
    """Test handling of invalid responses with a single example."""
    # Setup test data with pre-generated responses
    test_data = {"llm_responses": ["Invalid response"], "actual_labels": [0]}
    test_file = setup_test_data(tmp_path, test_data)

    # Mock completion to return invalid response
    mock_func = create_mock_completion(["INVALID_LABEL"])
    monkeypatch.setattr(
        "ferrari.together_code.fomc.fomc_evaluate.completion", mock_func
    )

    # Run evaluation
    try:
        results_df, _ = fomc_evaluate(str(test_file), task_args)
        assert (results_df["extracted_labels"] == -1).any()
    except Exception as e:
        pytest.fail(f"Invalid response handling failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
