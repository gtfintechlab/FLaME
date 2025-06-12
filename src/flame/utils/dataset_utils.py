"""Utilities for safe dataset loading with proper error handling."""

import os
import sys
from typing import Any, Optional

from datasets import load_dataset

from flame.utils.logging_utils import get_component_logger

logger = get_component_logger("utils.dataset")


def safe_load_dataset(
    dataset_name: str,
    split: Optional[str] = None,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Safely load a dataset with proper error handling for authentication issues.

    Args:
        dataset_name: The name of the dataset to load
        split: Optional split to load (e.g., "train", "test")
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments to pass to load_dataset

    Returns:
        The loaded dataset

    Raises:
        SystemExit: If authentication fails or dataset cannot be accessed
    """
    # In CI/test environments with mock token, return a mock dataset
    if (
        os.getenv("CI") == "true"
        and os.getenv("HUGGINGFACEHUB_API_TOKEN") == "mock-token-for-ci"
    ):
        logger.debug(f"CI mode: returning mock dataset for {dataset_name}")

        # Return a mock dataset that mimics HuggingFace dataset structure
        class MockDataset(list):
            def __init__(self):
                # Create some dummy data with all common fields
                super().__init__(
                    [
                        {
                            "text": "dummy text 1",
                            "label": 0,
                            "sentence": "Test sentence 1",
                            "question": "Test question 1?",
                            "pre_text": ["Pre text 1"],
                            "post_text": ["Post text 1"],
                            "table_ori": [["A", "B"], ["1", "2"]],
                            "question_0": "Q0",
                            "question_1": "Q1",
                            "answer_0": "A0",
                            "answer_1": "A1",
                            "context": "Test context 1",
                            "tokens": ["token1", "token2"],
                            "query": "Test query 1",
                            "narrative": "Test narrative 1",
                            "summary": "Test summary 1",
                            "answer": "Test answer 1",
                            "choices": ["choice1", "choice2", "choice3", "choice4"],
                            "company": "Test Company 1",
                            "docType": "10-K",
                            "numerals-tags": '{"100": "NUMBER", "2023": "DATE"}',
                            "response": "positive",
                            "tags": ["B-ORG", "O", "O"],
                        },
                        {
                            "text": "dummy text 2",
                            "label": 1,
                            "sentence": "Test sentence 2",
                            "question": "Test question 2?",
                            "pre_text": ["Pre text 2"],
                            "post_text": ["Post text 2"],
                            "table_ori": [["C", "D"], ["3", "4"]],
                            "question_0": "Q0",
                            "question_1": "Q1",
                            "answer_0": "A0",
                            "answer_1": "A1",
                            "context": "Test context 2",
                            "tokens": ["token3", "token4"],
                            "query": "Test query 2",
                            "narrative": "Test narrative 2",
                            "summary": "Test summary 2",
                            "answer": "Test answer 2",
                            "choices": ["choice1", "choice2", "choice3", "choice4"],
                            "company": "Test Company 2",
                            "docType": "8-K",
                            "numerals-tags": '{"200": "MONEY", "2024": "DATE"}',
                            "response": "negative",
                            "tags": ["O", "B-PER", "O"],
                        },
                    ]
                )

            def __getitem__(self, key):
                if key in {"train", "test", "validation", "dev"}:
                    return self
                return super().__getitem__(key)

        mock_dataset = MockDataset()
        if split:
            return mock_dataset[split]
        return mock_dataset

    try:
        logger.debug(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(
            dataset_name, trust_remote_code=trust_remote_code, **kwargs
        )

        if split:
            dataset = dataset[split]

        logger.debug(f"Successfully loaded dataset: {dataset_name}")
        return dataset

    except Exception as e:
        error_msg = str(e).lower()

        # Check for authentication-related errors
        if any(
            auth_err in error_msg
            for auth_err in [
                "401",
                "unauthorized",
                "forbidden",
                "authentication",
                "private",
                "token",
                "credential",
                "permission denied",
            ]
        ):
            logger.error(f"Authentication error loading dataset '{dataset_name}': {e}")
            print(
                f"\nERROR: Failed to load dataset '{dataset_name}' due to authentication issues."
            )
            print("Please ensure:")
            print("1. Your HUGGINGFACEHUB_API_TOKEN is set correctly")
            print("2. Your token has access to the required datasets")
            print("3. You are logged in to Hugging Face Hub")
            sys.exit(1)

        # Check for dataset not found errors
        elif any(
            not_found in error_msg
            for not_found in ["404", "not found", "does not exist", "couldn't find"]
        ):
            logger.error(f"Dataset not found: '{dataset_name}'")
            print(f"\nERROR: Dataset '{dataset_name}' not found.")
            print("Please check the dataset name is correct.")
            sys.exit(1)

        # Other errors
        else:
            logger.error(f"Error loading dataset '{dataset_name}': {e}")
            print(f"\nERROR: Failed to load dataset '{dataset_name}': {e}")
            sys.exit(1)
