"""Tests for MMLU inference functionality."""

import pytest
import pandas as pd
from pathlib import Path
from ferrari.code.mmlu.mmlu_inference import (
    format_mmlu_prompt,
    generate_inference_filename,
    mmlu_inference,
)
from ferrari.code.mmlu.mmlu_loader import MMLULoader
from ferrari.code.mmlu.mmlu_constants import ECONOMICS_SUBJECTS
from tests.utils.fixtures import create_mock_completion

# Reuse sample data from test_mmlu_evaluate.py
SAMPLE_QUESTIONS = [
    {
        "subject": "high_school_microeconomics",
        "question": "What is the law of demand?",
        "choices": [
            "As price increases, quantity demanded increases",
            "As price increases, quantity demanded decreases",
            "As price decreases, quantity demanded decreases",
            "Price and quantity demanded are unrelated",
        ],
        "answer": "B",
    },
    {
        "subject": "high_school_macroeconomics",
        "question": "What is GDP?",
        "choices": [
            "Total value of all final goods and services produced in a country",
            "Total value of all goods exported by a country",
            "Total value of all government spending",
            "Total value of all consumer spending",
        ],
        "answer": "A",
    },
]

SAMPLE_FEW_SHOT = [
    {
        "subject": "high_school_microeconomics",
        "question": "What happens to the demand curve when consumer income increases for a normal good?",
        "choices": ["Shifts left", "Shifts right", "Does not shift", "Becomes steeper"],
        "answer": "B",
    }
]


@pytest.fixture
def mmlu_args(base_args):
    """Create MMLU-specific test arguments."""
    base_args.dataset = "mmlu"
    base_args.mmlu_subjects = ECONOMICS_SUBJECTS
    base_args.mmlu_split = "test"
    base_args.mmlu_num_few_shot = 5
    base_args.batch_size = 2
    return base_args


def test_format_mmlu_prompt():
    """Test MMLU prompt formatting."""
    prompt = format_mmlu_prompt(
        SAMPLE_QUESTIONS[0]["question"],
        SAMPLE_QUESTIONS[0]["choices"],
        [SAMPLE_FEW_SHOT],
    )

    # Check prompt structure
    assert "Question:" in prompt
    assert "Choices:" in prompt
    assert "Answer:" in prompt

    # Check example inclusion
    assert SAMPLE_FEW_SHOT[0]["question"] in prompt
    assert SAMPLE_FEW_SHOT[0]["answer"] in prompt

    # Check current question
    assert SAMPLE_QUESTIONS[0]["question"] in prompt
    assert all(choice in prompt for choice in SAMPLE_QUESTIONS[0]["choices"])


def test_generate_inference_filename():
    """Test inference filename generation."""
    task = "mmlu"
    model = "together_ai/meta-llama/Llama-2-7b"

    base_name, full_path = generate_inference_filename(task, model)

    # Check filename format
    assert task in base_name
    assert "together_ai" in base_name
    assert "Llama" in base_name
    assert isinstance(full_path, Path)
    assert full_path.parent.name == task
    assert full_path.name.startswith("inference_")
    assert full_path.suffix == ".csv"


def test_batch_processing(mmlu_args, monkeypatch):
    """Test batch processing of questions."""
    # Mock the dataset loading
    mock_data = pd.DataFrame(SAMPLE_QUESTIONS)
    monkeypatch.setattr(MMLULoader, "load", lambda self: (mock_data, [SAMPLE_FEW_SHOT]))

    # Mock model responses
    mock_responses = ["B", "A"]  # Correct answers for both questions
    mock_func = create_mock_completion(mock_responses)
    monkeypatch.setattr(
        "ferrari.code.mmlu.mmlu_inference.litellm.batch_completion", mock_func
    )

    try:
        results_df = mmlu_inference(mmlu_args)

        # Verify results
        assert len(results_df) == len(SAMPLE_QUESTIONS)
        assert "raw_response" in results_df.columns
        assert "subject" in results_df.columns
        assert all(not pd.isna(resp) for resp in results_df["raw_response"])

    except Exception as e:
        pytest.fail(f"Batch processing failed: {str(e)}")


def test_error_handling(mmlu_args, monkeypatch):
    """Test handling of API errors and invalid responses."""
    # Mock dataset loading
    mock_data = pd.DataFrame(SAMPLE_QUESTIONS)
    monkeypatch.setattr(MMLULoader, "load", lambda self: (mock_data, [SAMPLE_FEW_SHOT]))

    # Mock API error for first batch
    def mock_api_error(*args, **kwargs):
        raise Exception("API Error")

    monkeypatch.setattr(
        "ferrari.code.mmlu.mmlu_inference.litellm.batch_completion", mock_api_error
    )

    try:
        results_df = mmlu_inference(mmlu_args)

        # Verify error handling
        assert len(results_df) == len(SAMPLE_QUESTIONS)
        assert all(pd.isna(resp) for resp in results_df["raw_response"])

    except Exception as e:
        pytest.fail(f"Error handling failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
