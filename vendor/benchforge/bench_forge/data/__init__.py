"""Data management module for BenchForge.

Professional-grade data handling with loaders, processors, splitters, and caching.
"""

# Import from loader
from .loader import (
    LoaderConfig,
    DatasetLoader,
    HuggingFaceLoader,
    JSONLoader,
    CSVLoader,
    ParquetLoader,
    loader_factory,
    load_dataset,
)

# Import from processor
from .processor import (
    ProcessorConfig,
    DataProcessor,
    TextProcessor,
    DatasetProcessor,
    process_dataset,
)

# Import from splitter
from .splitter import (
    SplitConfig,
    DataSplitter,
    train_test_split,
)

# Import from cache
from .cache import (
    CacheConfig,
    CacheManager,
    ResponseCache,
)

# Module exports
__all__ = [
    # Loader
    "LoaderConfig",
    "DatasetLoader",
    "HuggingFaceLoader",
    "JSONLoader",
    "CSVLoader",
    "ParquetLoader",
    "loader_factory",
    "load_dataset",
    # Processor
    "ProcessorConfig",
    "DataProcessor",
    "TextProcessor",
    "DatasetProcessor",
    "process_dataset",
    # Splitter
    "SplitConfig",
    "DataSplitter",
    "train_test_split",
    # Cache
    "CacheConfig",
    "CacheManager",
    "ResponseCache",
]
