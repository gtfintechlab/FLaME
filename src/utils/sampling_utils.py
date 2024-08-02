from datasets import Dataset


def sample_dataset(dataset: Dataset, sample_size: int, method: str):
    if method == "random":
        return dataset.shuffle(seed=42).select(range(sample_size))
    elif method == "head":
        return dataset.select(range(sample_size))
    elif method == "tail":
        return dataset.select(range(len(dataset) - sample_size, len(dataset)))
    else:
        raise ValueError("Method must be 'random', 'head', or 'tail'.")
