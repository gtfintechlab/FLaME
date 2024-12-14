"""Utilities for dataset sampling."""

from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from superflue.utils.logging_utils import get_logger

logger = get_logger(__name__)


# TODO: (Glenn) Adapt this function to be the standard dataset loader for all SuperFLUE tasks.
# Each task should migrate to using this function for dataset loading to ensure consistent
# sampling behavior across the entire codebase.
def load_and_sample_dataset(
    dataset_path: str,
    sample_size: int | None = None,
    sample_method: str = "head",
    split: str = "test",
    **kwargs,
) -> Dataset:
    """Load a dataset and optionally sample from it.

    Args:
        dataset_path: HuggingFace dataset path (e.g. 'org/dataset_name')
        sample_size: Number of samples to return. If None, returns full dataset
        sample_method: Sampling method - 'random', 'head', or 'tail'
        split: Dataset split to use
        **kwargs: Additional arguments passed to load_dataset

    Returns:
        Dataset: The loaded (and optionally sampled) dataset
    """
    logger.debug(f"Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path, **kwargs)

    # Get the requested split
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        dataset = dataset[split]

    # Return full dataset if no sampling requested
    if not sample_size:
        logger.debug(f"Using full dataset: {len(dataset)} samples")
        return dataset

    # Apply sampling
    if sample_method == "random":
        sampled = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
    elif sample_method == "head":
        sampled = dataset.select(range(min(sample_size, len(dataset))))
    elif sample_method == "tail":
        start_idx = max(0, len(dataset) - sample_size)
        sampled = dataset.select(range(start_idx, len(dataset)))
    else:
        raise ValueError("sample_method must be 'random', 'head', or 'tail'")

    logger.info(f"Sampled {len(sampled)} examples using method: {sample_method}")
    return sampled
