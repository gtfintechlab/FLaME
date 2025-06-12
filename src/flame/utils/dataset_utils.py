"""Utilities for safe dataset loading with proper error handling."""

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
