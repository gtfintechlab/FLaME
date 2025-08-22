"""Dataset splitting utilities for BenchForge.

Professional-grade data splitting with stratification, validation,
and multiple split strategies.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting."""

    random_state: int = 42
    shuffle: bool = True
    stratify: Optional[str] = None
    validate_splits: bool = True
    min_samples_per_split: int = 1
    allow_uneven: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.min_samples_per_split < 1:
            raise ValueError(
                f"min_samples_per_split must be positive, got {self.min_samples_per_split}"
            )


class DataSplitter:
    """Split datasets into train/validation/test sets with professional features."""

    def __init__(self, config: Optional[SplitConfig] = None):
        """Initialize splitter.

        Args:
            config: Split configuration
        """
        self.config = config or SplitConfig()
        self._stats = {
            "splits_created": 0,
            "total_samples_split": 0,
            "stratified_splits": 0,
            "validation_failures": 0,
        }

    def split(
        self,
        dataset: Union[List, pd.DataFrame],
        splits: Dict[str, float],
        random_state: Optional[int] = None,
        stratify: Optional[Union[List, str]] = None,
    ) -> Dict[str, Any]:
        """Split dataset into multiple sets with validation.

        Args:
            dataset: Dataset to split
            splits: Dictionary of split names and ratios (must sum to 1.0)
            random_state: Random seed (overrides config)
            stratify: Column/field for stratified splitting (overrides config)

        Returns:
            Dictionary of split datasets

        Raises:
            ValueError: If splits don't sum to 1.0 or other validation fails
        """
        # Validate splits
        self._validate_splits(splits)

        # Use provided or config values
        random_state = (
            random_state if random_state is not None else self.config.random_state
        )
        stratify = stratify if stratify is not None else self.config.stratify

        # Check dataset size
        n_samples = len(dataset)
        if n_samples < len(splits):
            raise ValueError(
                f"Dataset has {n_samples} samples, cannot create {len(splits)} splits"
            )

        # Check minimum samples per split
        min_split_size = min(int(n_samples * ratio) for ratio in splits.values())
        if min_split_size < self.config.min_samples_per_split:
            raise ValueError(
                f"Smallest split would have {min_split_size} samples, "
                f"minimum required is {self.config.min_samples_per_split}"
            )

        # Perform split based on type
        if isinstance(dataset, pd.DataFrame):
            result = self._split_dataframe(dataset, splits, random_state, stratify)
        elif isinstance(dataset, list):
            result = self._split_list(dataset, splits, random_state, stratify)
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

        # Validate result if configured
        if self.config.validate_splits:
            self._validate_result(result, splits, n_samples)

        # Update statistics
        self._stats["splits_created"] += len(splits)
        self._stats["total_samples_split"] += n_samples
        if stratify:
            self._stats["stratified_splits"] += 1

        return result

    def _validate_splits(self, splits: Dict[str, float]):
        """Validate split ratios.

        Args:
            splits: Split ratios

        Raises:
            ValueError: If validation fails
        """
        if not splits:
            raise ValueError("No splits provided")

        # Check all values are positive
        for name, ratio in splits.items():
            if ratio <= 0:
                raise ValueError(f"Split '{name}' has non-positive ratio: {ratio}")

        # Check sum
        total = sum(splits.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValueError(f"Splits must sum to 1.0, got {total}")

    def _split_dataframe(
        self,
        df: pd.DataFrame,
        splits: Dict[str, float],
        random_state: int,
        stratify: Optional[str],
    ) -> Dict[str, pd.DataFrame]:
        """Split pandas DataFrame with stratification support.

        Args:
            df: DataFrame to split
            splits: Split ratios
            random_state: Random seed
            stratify: Column name for stratification

        Returns:
            Dictionary of DataFrames
        """
        try:
            from sklearn.model_selection import train_test_split

            use_sklearn = True
        except ImportError:
            use_sklearn = False
            logger.warning("sklearn not available, using basic splitting")

        # Shuffle if configured
        if self.config.shuffle:
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        result = {}
        remaining = df
        remaining_ratio = 1.0

        # Get stratify column if specified
        stratify_col = None
        if stratify and stratify in df.columns:
            stratify_col = df[stratify]

            # Check if stratification is possible
            value_counts = stratify_col.value_counts()
            min_class_count = value_counts.min()

            if min_class_count < len(splits):
                logger.warning(
                    f"Cannot stratify: smallest class has {min_class_count} samples, "
                    f"need at least {len(splits)}"
                )
                stratify_col = None

        # Process splits in order
        split_names = list(splits.keys())

        for i, name in enumerate(split_names[:-1]):
            ratio = splits[name]
            # Adjust ratio based on remaining data
            adjusted_ratio = ratio / remaining_ratio

            if use_sklearn and stratify_col is not None:
                # Use sklearn for stratified split
                try:
                    current_stratify = (
                        remaining[stratify] if stratify in remaining.columns else None
                    )
                    current, remaining = train_test_split(
                        remaining,
                        test_size=(1 - adjusted_ratio),
                        random_state=random_state + i,
                        stratify=current_stratify,
                    )
                except ValueError as e:
                    # Stratification failed, fall back to regular split
                    logger.warning(f"Stratification failed: {e}, using regular split")
                    split_size = int(len(remaining) * adjusted_ratio)
                    current = remaining.iloc[:split_size]
                    remaining = remaining.iloc[split_size:]
            else:
                # Manual split
                split_size = int(len(remaining) * adjusted_ratio)
                current = remaining.iloc[:split_size]
                remaining = remaining.iloc[split_size:]

            result[name] = current
            remaining_ratio -= ratio

        # Last split gets remaining data
        result[split_names[-1]] = remaining

        # Log split information
        if self.config.verbose:
            for name, split_df in result.items():
                logger.info(
                    f"Split '{name}': {len(split_df)} samples "
                    f"({len(split_df) / len(df) * 100:.1f}%)"
                )

        return result

    def _split_list(
        self,
        dataset: List,
        splits: Dict[str, float],
        random_state: int,
        stratify: Optional[str],
    ) -> Dict[str, List]:
        """Split list dataset with optional stratification.

        Args:
            dataset: List to split
            splits: Split ratios
            random_state: Random seed
            stratify: Field name for stratification (if dict items)

        Returns:
            Dictionary of lists
        """
        import random

        random.seed(random_state)

        # Shuffle if configured
        if self.config.shuffle:
            dataset = dataset.copy()
            random.shuffle(dataset)

        # Handle stratification for list of dicts
        if stratify and dataset and isinstance(dataset[0], dict):
            return self._stratified_split_list(dataset, splits, random_state, stratify)

        # Regular split
        result = {}
        n = len(dataset)
        current_idx = 0

        # Process splits
        split_names = list(splits.keys())
        for i, name in enumerate(split_names[:-1]):
            ratio = splits[name]
            split_size = int(n * ratio)

            result[name] = dataset[current_idx : current_idx + split_size]
            current_idx += split_size

        # Last split gets remaining data
        result[split_names[-1]] = dataset[current_idx:]

        # Log split information
        if self.config.verbose:
            for name, split_data in result.items():
                logger.info(
                    f"Split '{name}': {len(split_data)} samples "
                    f"({len(split_data) / n * 100:.1f}%)"
                )

        return result

    def _stratified_split_list(
        self,
        dataset: List[Dict],
        splits: Dict[str, float],
        random_state: int,
        stratify_field: str,
    ) -> Dict[str, List]:
        """Perform stratified split on list of dictionaries.

        Args:
            dataset: List of dictionaries
            splits: Split ratios
            random_state: Random seed
            stratify_field: Field to stratify on

        Returns:
            Dictionary of stratified lists
        """
        import random

        random.seed(random_state)

        # Group by stratify field
        groups = {}
        for item in dataset:
            key = item.get(stratify_field, "__missing__")
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        # Check if stratification is feasible
        min_group_size = min(len(group) for group in groups.values())
        if min_group_size < len(splits):
            logger.warning(
                f"Cannot stratify: smallest group has {min_group_size} samples, "
                f"falling back to regular split"
            )
            return self._split_list(dataset, splits, random_state, None)

        # Initialize result
        result = {name: [] for name in splits.keys()}

        # Split each group
        for group_key, group_items in groups.items():
            # Shuffle group
            random.shuffle(group_items)

            # Split group according to ratios
            group_splits = self._split_list(
                group_items, splits, random_state + hash(group_key), None
            )

            # Add to results
            for split_name, split_items in group_splits.items():
                result[split_name].extend(split_items)

        # Shuffle final results
        for split_name in result:
            random.shuffle(result[split_name])

        return result

    def _validate_result(
        self,
        result: Dict[str, Any],
        expected_splits: Dict[str, float],
        total_samples: int,
    ):
        """Validate split results.

        Args:
            result: Split results
            expected_splits: Expected split ratios
            total_samples: Total number of samples

        Raises:
            ValueError: If validation fails
        """
        # Check all splits present
        if set(result.keys()) != set(expected_splits.keys()):
            missing = set(expected_splits.keys()) - set(result.keys())
            extra = set(result.keys()) - set(expected_splits.keys())
            raise ValueError(f"Split mismatch - missing: {missing}, extra: {extra}")

        # Check total samples preserved
        actual_total = sum(len(split) for split in result.values())
        if actual_total != total_samples:
            raise ValueError(
                f"Sample count mismatch: expected {total_samples}, got {actual_total}"
            )

        # Check ratios (with tolerance)
        if not self.config.allow_uneven:
            for name, expected_ratio in expected_splits.items():
                actual_ratio = len(result[name]) / total_samples
                if not np.isclose(actual_ratio, expected_ratio, rtol=0.05):
                    logger.warning(
                        f"Split '{name}' ratio mismatch: "
                        f"expected {expected_ratio:.3f}, got {actual_ratio:.3f}"
                    )

    def train_val_test_split(
        self,
        dataset: Union[List, pd.DataFrame],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: Optional[int] = None,
        stratify: Optional[Union[List, str]] = None,
    ) -> Tuple[Any, Any, Any]:
        """Convenience method for train/val/test split.

        Args:
            dataset: Dataset to split
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed
            stratify: Stratification column/field

        Returns:
            Tuple of (train, val, test) datasets
        """
        splits = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

        result = self.split(dataset, splits, random_state, stratify)
        return result["train"], result["val"], result["test"]

    def k_fold_split(
        self,
        dataset: Union[List, pd.DataFrame],
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ) -> List[Tuple[Any, Any]]:
        """Create k-fold cross-validation splits.

        Args:
            dataset: Dataset to split
            n_folds: Number of folds
            random_state: Random seed

        Returns:
            List of (train, val) tuples for each fold
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")

        random_state = random_state or self.config.random_state

        # Shuffle if configured
        if self.config.shuffle:
            if isinstance(dataset, pd.DataFrame):
                dataset = dataset.sample(frac=1, random_state=random_state).reset_index(
                    drop=True
                )
            else:
                import random

                random.seed(random_state)
                dataset = dataset.copy()
                random.shuffle(dataset)

        n_samples = len(dataset)
        fold_size = n_samples // n_folds
        remainder = n_samples % n_folds

        folds = []
        current_idx = 0

        for i in range(n_folds):
            # Add extra sample to first 'remainder' folds
            this_fold_size = fold_size + (1 if i < remainder else 0)

            if isinstance(dataset, pd.DataFrame):
                val_data = dataset.iloc[current_idx : current_idx + this_fold_size]
                train_data = pd.concat(
                    [
                        dataset.iloc[:current_idx],
                        dataset.iloc[current_idx + this_fold_size :],
                    ]
                )
            else:
                val_data = dataset[current_idx : current_idx + this_fold_size]
                train_data = (
                    dataset[:current_idx] + dataset[current_idx + this_fold_size :]
                )

            folds.append((train_data, val_data))
            current_idx += this_fold_size

        logger.info(f"Created {n_folds} cross-validation folds")
        return folds

    def get_stats(self) -> Dict[str, Any]:
        """Get splitter statistics.

        Returns:
            Statistics dictionary
        """
        return self._stats.copy()


# Convenience functions
def train_test_split(
    dataset: Union[List, pd.DataFrame],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[Union[List, str]] = None,
) -> Tuple[Any, Any]:
    """Simple train/test split.

    Args:
        dataset: Dataset to split
        test_size: Test set ratio
        random_state: Random seed
        stratify: Stratification column/field

    Returns:
        Tuple of (train, test) datasets
    """
    splitter = DataSplitter()
    splits = {"train": 1 - test_size, "test": test_size}
    result = splitter.split(dataset, splits, random_state, stratify)
    return result["train"], result["test"]


# Module exports
__all__ = [
    "SplitConfig",
    "DataSplitter",
    "train_test_split",
]
