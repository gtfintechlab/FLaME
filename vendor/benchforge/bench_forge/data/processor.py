"""Data preprocessing utilities for BenchForge.

Professional-grade data processing with transformations, filtering,
and dataset manipulation capabilities.
"""

import logging
import re
import string
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
import hashlib

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for data processing."""

    max_workers: int = 4
    batch_size: int = 1000
    cache_transformations: bool = True
    validate_output: bool = True
    error_handling: str = "raise"  # "raise", "skip", "default"
    default_value: Any = None
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.error_handling not in ["raise", "skip", "default"]:
            raise ValueError(f"Invalid error_handling: {self.error_handling}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


class DataProcessor:
    """Process and transform datasets with professional features."""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize processor.

        Args:
            config: Processor configuration
        """
        self.config = config or ProcessorConfig()
        self.transforms = []
        self._cache = {} if self.config.cache_transformations else None
        self._stats = {
            "transforms_applied": 0,
            "samples_processed": 0,
            "errors_encountered": 0,
            "cache_hits": 0,
        }

    def add_transform(
        self,
        transform: Callable[[Any], Any],
        name: Optional[str] = None,
        validate: Optional[Callable[[Any], bool]] = None,
    ) -> "DataProcessor":
        """Add a transformation function.

        Args:
            transform: Function to transform data
            name: Optional name for the transform
            validate: Optional validation function

        Returns:
            Self for chaining
        """
        transform_dict = {
            "function": transform,
            "name": name or transform.__name__,
            "validate": validate,
        }
        self.transforms.append(transform_dict)

        if self.config.verbose:
            logger.info(f"Added transform: {transform_dict['name']}")

        return self

    def process(self, data: Any) -> Any:
        """Apply all transforms to data with error handling.

        Args:
            data: Input data

        Returns:
            Transformed data

        Raises:
            ValueError: If processing fails and error_handling="raise"
        """
        result = data

        for transform in self.transforms:
            try:
                # Check cache if enabled
                if self._cache is not None:
                    cache_key = self._get_cache_key(result, transform["name"])
                    if cache_key in self._cache:
                        result = self._cache[cache_key]
                        self._stats["cache_hits"] += 1
                        continue

                # Apply transformation
                if self.config.verbose:
                    logger.debug(f"Applying transform: {transform['name']}")

                new_result = transform["function"](result)

                # Validate if provided
                if transform.get("validate") and not transform["validate"](new_result):
                    raise ValueError(
                        f"Validation failed for transform {transform['name']}"
                    )

                # Cache if enabled
                if self._cache is not None:
                    self._cache[cache_key] = new_result

                result = new_result
                self._stats["transforms_applied"] += 1

            except Exception as e:
                self._stats["errors_encountered"] += 1

                if self.config.error_handling == "raise":
                    raise ValueError(
                        f"Transform {transform['name']} failed: {e}"
                    ) from e
                elif self.config.error_handling == "skip":
                    logger.warning(f"Skipping transform {transform['name']}: {e}")
                    continue
                elif self.config.error_handling == "default":
                    logger.warning(
                        f"Using default for transform {transform['name']}: {e}"
                    )
                    result = self.config.default_value

        # Update stats
        if hasattr(result, "__len__"):
            self._stats["samples_processed"] = len(result)

        return result

    def _get_cache_key(self, data: Any, transform_name: str) -> str:
        """Generate cache key for transformation.

        Args:
            data: Input data
            transform_name: Transform name

        Returns:
            Cache key
        """
        # Create a hash of data representation
        if isinstance(data, (list, dict)):
            data_str = str(len(data))
        elif isinstance(data, pd.DataFrame):
            data_str = f"{data.shape}_{data.columns.tolist()}"
        else:
            data_str = str(type(data))

        key_str = f"{transform_name}:{data_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def reset(self):
        """Clear all transforms and cache."""
        self.transforms = []
        if self._cache is not None:
            self._cache.clear()
        logger.info("Processor reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics.

        Returns:
            Statistics dictionary
        """
        return self._stats.copy()

    def create_pipeline(self) -> Callable[[Any], Any]:
        """Create a callable pipeline from transforms.

        Returns:
            Pipeline function
        """

        def pipeline(data):
            return self.process(data)

        pipeline.__name__ = f"pipeline_{len(self.transforms)}_transforms"
        return pipeline


class TextProcessor:
    """Process text data with NLP-aware transformations."""

    # Precompiled patterns for efficiency
    _PATTERNS = {
        "whitespace": re.compile(r"\s+"),
        "urls": re.compile(r"https?://\S+|www\.\S+"),
        "emails": re.compile(r"\S+@\S+\.\S+"),
        "mentions": re.compile(r"@\w+"),
        "hashtags": re.compile(r"#\w+"),
        "numbers": re.compile(r"\d+"),
        "punctuation": re.compile(f"[{re.escape(string.punctuation)}]+"),
    }

    @staticmethod
    def clean(
        text: str,
        lowercase: bool = False,
        remove_extra_spaces: bool = True,
        strip: bool = True,
    ) -> str:
        """Clean text with configurable options.

        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_extra_spaces: Remove extra whitespace
            strip: Strip leading/trailing whitespace

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)

        if lowercase:
            text = text.lower()

        if remove_extra_spaces:
            text = TextProcessor._PATTERNS["whitespace"].sub(" ", text)

        if strip:
            text = text.strip()

        return text

    @staticmethod
    def truncate(
        text: str, max_length: int, strategy: str = "end", suffix: str = "..."
    ) -> str:
        """Truncate text with different strategies.

        Args:
            text: Input text
            max_length: Maximum length
            strategy: Truncation strategy ("end", "middle", "smart")
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        if strategy == "end":
            truncate_at = max_length - len(suffix)
            return text[:truncate_at] + suffix

        elif strategy == "middle":
            # Keep beginning and end
            suffix_len = len(suffix)
            keep_chars = max_length - suffix_len
            keep_start = keep_chars // 2
            keep_end = keep_chars - keep_start
            return text[:keep_start] + suffix + text[-keep_end:]

        elif strategy == "smart":
            # Try to truncate at word boundary
            truncate_at = max_length - len(suffix)
            truncated = text[:truncate_at]

            # Find last complete word
            last_space = truncated.rfind(" ")
            if last_space > truncate_at * 0.8:  # If we're not losing too much
                truncated = truncated[:last_space]

            return truncated + suffix

        else:
            raise ValueError(f"Unknown truncation strategy: {strategy}")

    @staticmethod
    def normalize(
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_extra_spaces: bool = True,
    ) -> str:
        """Normalize text with multiple options.

        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_numbers: Remove numbers
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_extra_spaces: Remove extra whitespace

        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            text = str(text)

        # Apply removals in order
        if remove_urls:
            text = TextProcessor._PATTERNS["urls"].sub("", text)

        if remove_emails:
            text = TextProcessor._PATTERNS["emails"].sub("", text)

        if remove_mentions:
            text = TextProcessor._PATTERNS["mentions"].sub("", text)

        if remove_hashtags:
            text = TextProcessor._PATTERNS["hashtags"].sub("", text)

        if remove_numbers:
            text = TextProcessor._PATTERNS["numbers"].sub("", text)

        if remove_punctuation:
            text = TextProcessor._PATTERNS["punctuation"].sub(" ", text)

        if lowercase:
            text = text.lower()

        if remove_extra_spaces:
            text = TextProcessor._PATTERNS["whitespace"].sub(" ", text).strip()

        return text

    @staticmethod
    def extract_features(text: str) -> Dict[str, Any]:
        """Extract features from text.

        Args:
            text: Input text

        Returns:
            Feature dictionary
        """
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "char_count": len(text.replace(" ", "")),
            "avg_word_length": np.mean([len(w) for w in text.split()])
            if text.split()
            else 0,
            "has_urls": bool(TextProcessor._PATTERNS["urls"].search(text)),
            "has_emails": bool(TextProcessor._PATTERNS["emails"].search(text)),
            "has_mentions": bool(TextProcessor._PATTERNS["mentions"].search(text)),
            "has_hashtags": bool(TextProcessor._PATTERNS["hashtags"].search(text)),
            "num_sentences": text.count(".") + text.count("!") + text.count("?"),
            "unique_words": len(set(text.lower().split())),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text)
            if text
            else 0,
        }


class DatasetProcessor:
    """Process entire datasets with batch operations."""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize dataset processor.

        Args:
            config: Processor configuration
        """
        self.config = config or ProcessorConfig()
        self._stats = {
            "datasets_processed": 0,
            "total_samples": 0,
            "filtered_samples": 0,
            "duplicates_removed": 0,
        }

    def filter_by_condition(
        self,
        dataset: Union[List[Dict], pd.DataFrame],
        condition: Callable[[Any], bool],
        invert: bool = False,
    ) -> Union[List[Dict], pd.DataFrame]:
        """Filter dataset by arbitrary condition.

        Args:
            dataset: Dataset to filter
            condition: Filter condition function
            invert: Invert the condition

        Returns:
            Filtered dataset
        """
        original_size = len(dataset)

        if isinstance(dataset, pd.DataFrame):
            mask = dataset.apply(condition, axis=1)
            if invert:
                mask = ~mask
            filtered = dataset[mask].copy()
        else:
            filtered = []
            for item in dataset:
                passes = condition(item)
                if (passes and not invert) or (not passes and invert):
                    filtered.append(item)

        filtered_size = len(filtered)
        self._stats["filtered_samples"] += original_size - filtered_size

        if self.config.verbose:
            logger.info(f"Filtered: {original_size} -> {filtered_size} samples")

        return filtered

    def filter_by_length(
        self,
        dataset: Union[List[Dict], pd.DataFrame],
        field: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Union[List[Dict], pd.DataFrame]:
        """Filter dataset by text length.

        Args:
            dataset: Dataset to filter
            field: Field name to check
            min_length: Minimum length
            max_length: Maximum length

        Returns:
            Filtered dataset
        """

        def length_condition(item):
            if isinstance(item, dict):
                value = item.get(field, "")
            else:
                value = item[field] if field in item else ""

            text_len = len(str(value))

            if min_length is not None and text_len < min_length:
                return False
            if max_length is not None and text_len > max_length:
                return False
            return True

        return self.filter_by_condition(dataset, length_condition)

    def sample(
        self,
        dataset: Union[List, pd.DataFrame],
        n: Optional[int] = None,
        frac: Optional[float] = None,
        stratify: Optional[str] = None,
        random_state: int = 42,
    ) -> Union[List, pd.DataFrame]:
        """Sample from dataset with stratification support.

        Args:
            dataset: Input dataset
            n: Number of samples
            frac: Fraction of samples
            stratify: Column/field for stratified sampling
            random_state: Random seed

        Returns:
            Sampled dataset
        """
        if isinstance(dataset, pd.DataFrame):
            if stratify and stratify in dataset.columns:
                # Stratified sampling for DataFrame
                from sklearn.model_selection import train_test_split

                _, sampled = train_test_split(
                    dataset,
                    test_size=(n / len(dataset) if n else frac),
                    stratify=dataset[stratify],
                    random_state=random_state,
                )
                return sampled
            else:
                return dataset.sample(n=n, frac=frac, random_state=random_state)

        elif isinstance(dataset, list):
            import random

            random.seed(random_state)

            if n:
                n = min(n, len(dataset))
                return random.sample(dataset, n)
            elif frac:
                n = int(len(dataset) * frac)
                return random.sample(dataset, n)
            else:
                return dataset

        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    def deduplicate(
        self,
        dataset: Union[List[Dict], pd.DataFrame],
        key_field: Optional[Union[str, List[str]]] = None,
        keep: str = "first",
    ) -> Union[List[Dict], pd.DataFrame]:
        """Remove duplicates from dataset.

        Args:
            dataset: Dataset to deduplicate
            key_field: Field(s) to check for duplicates
            keep: Which duplicate to keep ("first", "last", False)

        Returns:
            Deduplicated dataset
        """
        original_size = len(dataset)

        if isinstance(dataset, pd.DataFrame):
            if key_field:
                deduped = dataset.drop_duplicates(subset=key_field, keep=keep)
            else:
                deduped = dataset.drop_duplicates(keep=keep)

        else:  # List of dicts
            if key_field is None:
                # Use all fields
                seen = set()
                deduped = []

                for item in dataset:
                    item_key = str(sorted(item.items()))
                    if item_key not in seen:
                        seen.add(item_key)
                        deduped.append(item)

            else:
                # Use specific field(s)
                seen = set()
                deduped = []

                if isinstance(key_field, str):
                    key_field = [key_field]

                for item in dataset:
                    item_key = tuple(item.get(k) for k in key_field)
                    if item_key not in seen:
                        seen.add(item_key)
                        deduped.append(item)

        duplicates_removed = original_size - len(deduped)
        self._stats["duplicates_removed"] += duplicates_removed

        if self.config.verbose:
            logger.info(
                f"Deduplicated: {original_size} -> {len(deduped)} samples ({duplicates_removed} removed)"
            )

        return deduped

    def balance_classes(
        self,
        dataset: Union[List[Dict], pd.DataFrame],
        label_field: str,
        strategy: str = "undersample",
        random_state: int = 42,
    ) -> Union[List[Dict], pd.DataFrame]:
        """Balance classes in dataset.

        Args:
            dataset: Dataset to balance
            label_field: Label field name
            strategy: Balancing strategy ("undersample", "oversample")
            random_state: Random seed

        Returns:
            Balanced dataset
        """
        if isinstance(dataset, pd.DataFrame):
            # Get class counts
            class_counts = dataset[label_field].value_counts()

            if strategy == "undersample":
                min_count = class_counts.min()
                balanced_dfs = []

                for label in class_counts.index:
                    class_df = dataset[dataset[label_field] == label]
                    sampled = class_df.sample(n=min_count, random_state=random_state)
                    balanced_dfs.append(sampled)

                balanced = pd.concat(balanced_dfs, ignore_index=True)

            elif strategy == "oversample":
                max_count = class_counts.max()
                balanced_dfs = []

                for label in class_counts.index:
                    class_df = dataset[dataset[label_field] == label]
                    n_samples = max_count - len(class_df)

                    if n_samples > 0:
                        oversampled = class_df.sample(
                            n=n_samples, replace=True, random_state=random_state
                        )
                        combined = pd.concat([class_df, oversampled], ignore_index=True)
                        balanced_dfs.append(combined)
                    else:
                        balanced_dfs.append(class_df)

                balanced = pd.concat(balanced_dfs, ignore_index=True)

            else:
                raise ValueError(f"Unknown balancing strategy: {strategy}")

            # Shuffle the result
            balanced = balanced.sample(frac=1, random_state=random_state).reset_index(
                drop=True
            )

        else:  # List of dicts
            import random

            random.seed(random_state)

            # Group by label
            label_groups = {}
            for item in dataset:
                label = item.get(label_field)
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(item)

            if strategy == "undersample":
                min_count = min(len(group) for group in label_groups.values())
                balanced = []

                for label, items in label_groups.items():
                    sampled = random.sample(items, min_count)
                    balanced.extend(sampled)

            elif strategy == "oversample":
                max_count = max(len(group) for group in label_groups.values())
                balanced = []

                for label, items in label_groups.items():
                    if len(items) < max_count:
                        # Oversample with replacement
                        oversampled = random.choices(items, k=max_count)
                        balanced.extend(oversampled)
                    else:
                        balanced.extend(items)

            else:
                raise ValueError(f"Unknown balancing strategy: {strategy}")

            # Shuffle
            random.shuffle(balanced)

        logger.info(f"Balanced dataset: {len(dataset)} -> {len(balanced)} samples")
        return balanced

    def apply_transform(
        self,
        dataset: Union[List[Dict], pd.DataFrame],
        field: str,
        transform: Callable[[Any], Any],
        new_field: Optional[str] = None,
    ) -> Union[List[Dict], pd.DataFrame]:
        """Apply transformation to specific field.

        Args:
            dataset: Dataset to transform
            field: Field to transform
            transform: Transformation function
            new_field: Optional new field name for result

        Returns:
            Transformed dataset
        """
        output_field = new_field or field

        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.copy()
            dataset[output_field] = dataset[field].apply(transform)
        else:
            for item in dataset:
                if field in item:
                    item[output_field] = transform(item[field])

        self._stats["datasets_processed"] += 1
        self._stats["total_samples"] += len(dataset)

        return dataset

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics.

        Returns:
            Statistics dictionary
        """
        return self._stats.copy()


def process_dataset(
    dataset: List[Dict[str, Any]],
    config: Optional[ProcessorConfig] = None,
    text_field: str = "text",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Process a dataset with the specified configuration.

    Args:
        dataset: Dataset to process
        config: Processor configuration
        text_field: Field containing text to process
        **kwargs: Additional processing parameters

    Returns:
        Processed dataset
    """
    if config is None:
        config = ProcessorConfig()

    processor = DatasetProcessor(config)
    return processor.process(dataset, text_field=text_field, **kwargs)


# Module exports
__all__ = [
    "ProcessorConfig",
    "DataProcessor",
    "TextProcessor",
    "DatasetProcessor",
    "process_dataset",
]
