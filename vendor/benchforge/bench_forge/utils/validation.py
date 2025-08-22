"""Validation utilities for BenchForge.

Professional validation system for inputs, outputs, and configurations.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Validation error exception with detailed information."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        """Initialize validation error.

        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
        """
        self.field = field
        self.value = value
        super().__init__(message)

        # Log the error
        logger.error(f"Validation error: {message} (field: {field}, value: {value})")


@dataclass
class ValidationRule:
    """Validation rule definition."""

    name: str
    validator: Callable[[Any], bool]
    message: str
    severity: str = "error"  # error, warning, info

    def validate(self, value: Any) -> bool:
        """Apply validation rule.

        Args:
            value: Value to validate

        Returns:
            True if valid
        """
        try:
            return self.validator(value)
        except Exception as e:
            logger.debug(f"Validation rule {self.name} failed: {e}")
            return False


@dataclass
class ValidationResult:
    """Result of validation process."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str):
        """Add error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """Add warning message."""
        self.warnings.append(message)

    def add_info(self, message: str):
        """Add info message."""
        self.info.append(message)

    def raise_if_invalid(self):
        """Raise ValidationError if invalid."""
        if not self.is_valid:
            error_msg = "; ".join(self.errors)
            raise ValidationError(f"Validation failed: {error_msg}")


class InputValidator:
    """Validate inputs for BenchForge operations."""

    def __init__(self, strict: bool = False):
        """Initialize validator.

        Args:
            strict: Whether to fail on warnings
        """
        self.strict = strict
        self._stats = {
            "validations_performed": 0,
            "validations_passed": 0,
            "validations_failed": 0,
        }

    def validate_dataset(
        self,
        dataset: Any,
        min_size: int = 1,
        max_size: Optional[int] = None,
        required_fields: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate dataset format and content.

        Args:
            dataset: Dataset to validate
            min_size: Minimum size
            max_size: Maximum size
            required_fields: Required fields for dict datasets

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        self._stats["validations_performed"] += 1

        # Check if dataset is None
        if dataset is None:
            result.add_error("Dataset is None")
            self._stats["validations_failed"] += 1
            return result

        # Get dataset size
        try:
            dataset_size = len(dataset)
        except TypeError:
            result.add_error(f"Dataset type {type(dataset)} has no length")
            self._stats["validations_failed"] += 1
            return result

        # Check size constraints
        if dataset_size < min_size:
            result.add_error(f"Dataset size {dataset_size} < minimum {min_size}")

        if max_size and dataset_size > max_size:
            result.add_error(f"Dataset size {dataset_size} > maximum {max_size}")

        # Type-specific validation
        if isinstance(dataset, pd.DataFrame):
            self._validate_dataframe(dataset, result, required_fields)
        elif isinstance(dataset, list):
            self._validate_list(dataset, result, required_fields)
        elif hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
            # HuggingFace-style dataset
            result.add_info("Dataset appears to be HuggingFace format")
            if required_fields and dataset_size > 0:
                sample = dataset[0]
                missing = [f for f in required_fields if f not in sample]
                if missing:
                    result.add_error(f"Missing required fields: {missing}")
        else:
            result.add_warning(f"Unknown dataset type: {type(dataset)}")

        # Update stats
        if result.is_valid:
            self._stats["validations_passed"] += 1
        else:
            self._stats["validations_failed"] += 1

        return result

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
        required_fields: Optional[List[str]],
    ):
        """Validate pandas DataFrame."""
        # Check for required columns
        if required_fields:
            missing = [f for f in required_fields if f not in df.columns]
            if missing:
                result.add_error(f"Missing required columns: {missing}")

        # Check for empty DataFrame
        if df.empty:
            result.add_warning("DataFrame is empty")

        # Check for NaN-only columns
        nan_cols = df.columns[df.isna().all()].tolist()
        if nan_cols:
            result.add_warning(f"Columns with only NaN values: {nan_cols}")

        # Add metadata
        result.metadata["shape"] = df.shape
        result.metadata["columns"] = df.columns.tolist()
        result.metadata["dtypes"] = {
            col: str(dtype) for col, dtype in df.dtypes.items()
        }

    def _validate_list(
        self, data: list, result: ValidationResult, required_fields: Optional[List[str]]
    ):
        """Validate list dataset."""
        if not data:
            result.add_warning("List is empty")
            return

        # Check if all items are dicts
        non_dict_indices = [
            i for i, item in enumerate(data[:10]) if not isinstance(item, dict)
        ]
        if non_dict_indices:
            result.add_error(f"Non-dict items at indices: {non_dict_indices}")
            return

        # Check required fields
        if required_fields and isinstance(data[0], dict):
            sample_missing = []
            for i, item in enumerate(data[:5]):  # Check first 5
                missing = [f for f in required_fields if f not in item]
                if missing:
                    sample_missing.append((i, missing))

            if sample_missing:
                result.add_error(f"Missing fields in samples: {sample_missing}")

    def validate_prompt(
        self, prompt: str, max_length: int = 100000, min_length: int = 1
    ) -> ValidationResult:
        """Validate prompt format and content.

        Args:
            prompt: Prompt to validate
            max_length: Maximum length
            min_length: Minimum length

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if not isinstance(prompt, str):
            result.add_error(f"Prompt must be string, got {type(prompt)}")
            return result

        prompt_length = len(prompt)

        if prompt_length < min_length:
            result.add_error(f"Prompt too short: {prompt_length} < {min_length}")

        if prompt_length > max_length:
            result.add_error(f"Prompt too long: {prompt_length} > {max_length}")

        # Check for empty/whitespace-only
        if not prompt.strip():
            result.add_error("Prompt is empty or whitespace-only")

        # Add metadata
        result.metadata["length"] = prompt_length
        result.metadata["word_count"] = len(prompt.split())
        result.metadata["line_count"] = prompt.count("\n") + 1

        return result

    def validate_response(
        self, response: str, expected_format: Optional[str] = None
    ) -> ValidationResult:
        """Validate model response.

        Args:
            response: Response to validate
            expected_format: Expected format (text, json, etc.)

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if not isinstance(response, str):
            result.add_error(f"Response must be string, got {type(response)}")
            return result

        # Empty responses are allowed but flagged
        if not response:
            result.add_warning("Response is empty")

        # Format-specific validation
        if expected_format == "json":
            import json

            try:
                json.loads(response)
                result.add_info("Valid JSON response")
            except json.JSONDecodeError as e:
                result.add_error(f"Invalid JSON: {e}")

        elif expected_format == "number":
            try:
                float(response.strip())
                result.add_info("Valid numeric response")
            except ValueError:
                result.add_error(f"Expected number, got: {response[:50]}")

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics.

        Returns:
            Statistics dictionary
        """
        stats = self._stats.copy()
        if stats["validations_performed"] > 0:
            stats["success_rate"] = (
                stats["validations_passed"] / stats["validations_performed"]
            )
        else:
            stats["success_rate"] = 0.0
        return stats


class OutputValidator:
    """Validate outputs from BenchForge operations."""

    def validate_results(
        self, results: Any, required_columns: Optional[List[str]] = None
    ) -> ValidationResult:
        """Validate results format.

        Args:
            results: Results to validate
            required_columns: Required columns for DataFrame results

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Default required columns
        if required_columns is None:
            required_columns = ["input", "prompt", "raw_response", "extracted_response"]

        # Check DataFrame results
        if isinstance(results, pd.DataFrame):
            missing = [col for col in required_columns if col not in results.columns]
            if missing:
                result.add_error(f"Missing required columns: {missing}")

            if results.empty:
                result.add_warning("Results DataFrame is empty")

            result.metadata["shape"] = results.shape
            result.metadata["columns"] = results.columns.tolist()

        # Check list results
        elif isinstance(results, list):
            if not results:
                result.add_warning("Results list is empty")
            elif isinstance(results[0], dict):
                # Check first few items for required keys
                for i, item in enumerate(results[:5]):
                    missing = [key for key in required_columns if key not in item]
                    if missing:
                        result.add_error(f"Result {i} missing keys: {missing}")
                        break

        else:
            result.add_error(f"Unsupported results type: {type(results)}")

        return result

    def validate_metrics(
        self, metrics: Dict[str, float], expected_range: tuple = (0.0, 1.0)
    ) -> ValidationResult:
        """Validate metrics format and values.

        Args:
            metrics: Metrics to validate
            expected_range: Expected value range

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if not isinstance(metrics, dict):
            result.add_error(f"Metrics must be dictionary, got {type(metrics)}")
            return result

        for name, value in metrics.items():
            if not isinstance(name, str):
                result.add_error(f"Metric name must be string, got {type(name)}")
                continue

            if value is None:
                result.add_warning(f"Metric '{name}' is None")
                continue

            if not isinstance(value, (int, float)):
                result.add_error(f"Metric '{name}' must be numeric, got {type(value)}")
                continue

            # Check range
            if not (expected_range[0] <= value <= expected_range[1]):
                result.add_warning(
                    f"Metric '{name}' value {value} outside expected range {expected_range}"
                )

        result.metadata["num_metrics"] = len(metrics)
        result.metadata["metric_names"] = list(metrics.keys())

        return result


class ConfigValidator:
    """Validate configuration values."""

    @staticmethod
    def validate_task_config(config: Any) -> ValidationResult:
        """Validate task configuration.

        Args:
            config: Configuration to validate

        Returns:
            ValidationResult
        """
        from bench_forge.tasks.config import TaskConfig

        result = ValidationResult(is_valid=True)

        if not isinstance(config, TaskConfig):
            result.add_error(f"Config must be TaskConfig, got {type(config)}")
            return result

        # Validate required fields
        if not config.name:
            result.add_error("Task name is required")

        if not config.dataset:
            result.add_error("Dataset is required")

        # Validate ranges
        if not (0 <= config.temperature <= 2):
            result.add_error(f"Temperature must be [0, 2], got {config.temperature}")

        if not (0 <= config.top_p <= 1):
            result.add_error(f"top_p must be [0, 1], got {config.top_p}")

        if config.batch_size < 1:
            result.add_error(f"Batch size must be positive, got {config.batch_size}")

        if config.max_tokens < 1:
            result.add_error(f"Max tokens must be positive, got {config.max_tokens}")

        return result


# Convenience functions
def validate_dataset(dataset: Any, **kwargs) -> ValidationResult:
    """Validate dataset.

    Args:
        dataset: Dataset to validate
        **kwargs: Additional validation parameters

    Returns:
        ValidationResult
    """
    validator = InputValidator()
    return validator.validate_dataset(dataset, **kwargs)


def validate_prompt(prompt: str, **kwargs) -> ValidationResult:
    """Validate prompt.

    Args:
        prompt: Prompt to validate
        **kwargs: Additional validation parameters

    Returns:
        ValidationResult
    """
    validator = InputValidator()
    return validator.validate_prompt(prompt, **kwargs)


# Module exports
__all__ = [
    "ValidationError",
    "ValidationRule",
    "ValidationResult",
    "InputValidator",
    "OutputValidator",
    "ConfigValidator",
    "validate_dataset",
    "validate_prompt",
]
