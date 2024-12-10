"""Common test fixtures and utilities for Ferrari tests."""
import pytest
import pandas as pd
import numpy as np
from argparse import Namespace
from typing import Dict, Any
from pathlib import Path

@pytest.fixture
def base_args() -> Namespace:
    """Common base arguments used across different tasks."""
    return Namespace(
        dataset="mmlu",
        model="together_ai/meta-llama/Llama-2-7b",
        max_tokens=32,
        temperature=0.0,
        top_p=0.9,
        top_k=None,
        repetition_penalty=1.0,
        batch_size=10,
        mode="inference"
    )

class MockMessage:
    """Mock message object that mimics litellm response structure."""
    def __init__(self, content: str):
        self.content = content

class MockChoice:
    """Mock choice object that mimics litellm response structure."""
    def __init__(self, message: MockMessage):
        self.message = message
        self.finish_reason = "stop"

class MockResponse:
    """Mock response object that mimics litellm completion response."""
    def __init__(self, content: str):
        self.choices = [MockChoice(MockMessage(content))]

def create_mock_completion(responses: list) -> callable:
    """Create a mock completion function that returns responses in sequence.
    
    Args:
        responses: List of response strings to cycle through
        
    Returns:
        Mock completion function that returns MockResponse objects
    """
    if isinstance(responses, str):
        responses = [responses]
    
    def mock_batch_completion(*args, **kwargs) -> list:
        """Mock batch completion function."""
        return [MockResponse(resp) for resp in responses]
    
    return mock_batch_completion

def setup_test_data(tmp_path: Path, data: Dict[str, Any]) -> Path:
    """Set up test data in a temporary CSV file.
    
    Args:
        tmp_path: Temporary directory path from pytest
        data: Dictionary of data to save
        
    Returns:
        Path to created test file
    """
    df = pd.DataFrame(data)
    test_file = tmp_path / "test_data.csv"
    df.to_csv(test_file, index=False)
    return test_file

def assert_metrics_in_range(metrics_df: pd.DataFrame, metric_name: str, 
                          expected: float, tolerance: float = 1e-6) -> None:
    """Assert that a metric value is within expected range.
    
    Args:
        metrics_df: DataFrame containing metrics
        metric_name: Name of metric to check
        expected: Expected value
        tolerance: Allowed deviation from expected value
    """
    actual = metrics_df.loc[metrics_df["Metric"] == metric_name, "Value"].iloc[0]
    np.testing.assert_allclose(actual, expected, rtol=tolerance)

def mock_invalid_responses(monkeypatch, target_module: str) -> None:
    """Set up mock for invalid responses."""
    mock_func = create_mock_completion(["INVALID_RESPONSE"])
    monkeypatch.setattr(f"{target_module}.completion", mock_func) 