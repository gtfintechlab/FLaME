"""Constants and test data used across Ferrari tests."""

from typing import Dict, Any

# Test cases for different tasks - using minimal examples
FOMC_TEST_CASES = [
    {
        "llm_responses": ["Dovish response"],  # Single example for perfect accuracy
        "actual_labels": [0],
        "mock_responses": ["DOVISH"],
        "expected_accuracy": 1.0,
    },
    {
        "llm_responses": ["Mixed response"],  # Single example for zero accuracy
        "actual_labels": [0],
        "mock_responses": ["HAWKISH"],
        "expected_accuracy": 0.0,
    },
]

# Label mappings for different tasks
LABEL_MAPPINGS: Dict[str, Dict[str, int]] = {
    "fomc": {"DOVISH": 0, "HAWKISH": 1, "NEUTRAL": 2}
}

# Invalid response test data - minimal examples
INVALID_RESPONSE_DATA = {
    "fomc": {
        "llm_responses": ["Invalid"],  # Single invalid example
        "actual_labels": [0],
    }
}

# Minimal test data for file handling tests
MINIMAL_TEST_DATA: Dict[str, Dict[str, Any]] = {
    "fomc": {"llm_responses": ["Test response"], "actual_labels": [0]}
}
