"""Tests for MMLU evaluation functionality."""
import pytest
import pandas as pd
from pathlib import Path
from ferrari.code.mmlu.mmlu_evaluate import (
    extract_answer,
    mmlu_evaluate,
    validate_input_data
)
from tests.utils.fixtures import base_args, assert_metrics_in_range

# Sample test data
SAMPLE_RESULTS = pd.DataFrame({
    'question': [
        'What is the law of demand?',
        'What is GDP?',
        'What is inflation?',
        'What is heteroskedasticity?'
    ],
    'raw_response': [
        'The answer is B',
        'A is correct',
        'Let me think... C',
        'I am not sure about this one'
    ],
    'actual_answer': ['B', 'A', 'C', 'C'],
    'subject': [
        'high_school_microeconomics',
        'high_school_macroeconomics',
        'high_school_macroeconomics',
        'econometrics'
    ]
})

@pytest.fixture
def mmlu_args(base_args):
    """Create MMLU-specific test arguments."""
    base_args.dataset = "mmlu"
    base_args.model = "together_ai/meta-llama/Llama-2-7b"
    return base_args

def test_extract_answer():
    """Test answer extraction from model responses."""
    # Test valid answers
    assert extract_answer("A") == "A"
    assert extract_answer("The answer is B") == "B"
    assert extract_answer("C is correct") == "C"
    assert extract_answer("D.") == "D"
    
    # Test invalid answers
    assert extract_answer("E") is None
    assert extract_answer("Invalid") is None
    assert extract_answer("Let me think about it") is None
    assert extract_answer("") is None

def test_validate_input_data():
    """Test input data validation."""
    # Valid DataFrame
    valid_df = pd.DataFrame({
        'raw_response': ['A'],
        'actual_answer': ['B'],
        'subject': ['test']
    })
    validate_input_data(valid_df)  # Should not raise
    
    # Missing columns
    invalid_df = pd.DataFrame({
        'raw_response': ['A'],
        'actual_answer': ['B']
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input_data(invalid_df)

def test_evaluation_basic(mmlu_args, tmp_path):
    """Test basic evaluation functionality."""
    # Create a temporary results file
    results_file = tmp_path / "test_results.csv"
    SAMPLE_RESULTS.to_csv(results_file, index=False)
    
    try:
        results_df, metrics_df = mmlu_evaluate(str(results_file), mmlu_args)
        
        # Check results DataFrame
        assert len(results_df) == len(SAMPLE_RESULTS)
        assert "predicted_answer" in results_df.columns
        assert "actual_answer" in results_df.columns
        
        # Check metrics DataFrame
        assert len(metrics_df) > 0
        assert "Metric" in metrics_df.columns
        assert "Value" in metrics_df.columns
        
        # Check overall accuracy (3/4 = 0.75)
        assert_metrics_in_range(metrics_df, "Accuracy", 0.75, tolerance=0.01)
        
    except Exception as e:
        pytest.fail(f"Basic evaluation failed: {str(e)}")

def test_evaluation_all_invalid(mmlu_args, tmp_path):
    """Test evaluation with all invalid responses."""
    invalid_results = SAMPLE_RESULTS.copy()
    invalid_results['raw_response'] = "I am not sure"
    
    # Save to temp file
    results_file = tmp_path / "invalid_results.csv"
    invalid_results.to_csv(results_file, index=False)
    
    try:
        results_df, metrics_df = mmlu_evaluate(str(results_file), mmlu_args)
        
        # Check that all predictions are NaN
        assert all(pd.isna(pred) for pred in results_df["predicted_answer"])
        
        # Check metrics reflect invalid data
        assert metrics_df.loc[0, "Value"] == 0.0
        
    except Exception as e:
        pytest.fail(f"Invalid response handling failed: {str(e)}")

def test_evaluation_by_subject(mmlu_args, tmp_path):
    """Test evaluation metrics by subject."""
    results_file = tmp_path / "subject_results.csv"
    SAMPLE_RESULTS.to_csv(results_file, index=False)
    
    try:
        results_df, metrics_df = mmlu_evaluate(str(results_file), mmlu_args)
        
        # Check subject-specific metrics
        subject_metrics = metrics_df[metrics_df["Metric"].str.contains("_Accuracy")]
        
        # Should have metrics for each subject
        unique_subjects = SAMPLE_RESULTS["subject"].unique()
        assert len(subject_metrics) == len(unique_subjects)
        
        # Each subject should have a value between 0 and 1
        assert all(0 <= val <= 1 for val in subject_metrics["Value"])
        
    except Exception as e:
        pytest.fail(f"Subject-specific evaluation failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])