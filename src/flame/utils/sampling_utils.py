# TODO: (Glenn) One function for one file again. Can be refactored into another file
from datasets import Dataset, DatasetDict, IterableDatasetDict


def sample_dataset(dataset, sample_size: int, method: str, split: str = "train"):
    # Handle different dataset types
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]  # Adjust if you need a different split
    elif isinstance(dataset, IterableDatasetDict):
        dataset = dataset[split]  # Adjust if needed

    # Ensure dataset is a Dataset type
    if not isinstance(dataset, Dataset):
        raise TypeError("Expected dataset to be of type 'Dataset'.")

    if method == "random":
        return dataset.shuffle(seed=42).select(range(sample_size))
    elif method == "head":
        return dataset.select(range(sample_size))
    elif method == "tail":
        return dataset.select(range(len(dataset) - sample_size, len(dataset)))
    else:
        raise ValueError("Method must be 'random', 'head', or 'tail'.")
