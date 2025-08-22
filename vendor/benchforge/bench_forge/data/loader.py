"""Dataset loading abstraction for BenchForge.

Professional-grade data loading system with multiple format support,
validation, and extensible factory pattern.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass
from functools import lru_cache
import hashlib

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for dataset loaders."""

    cache_dir: Optional[Path] = None
    validate_on_load: bool = True
    allow_empty: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    trust_remote_code: bool = False
    streaming: bool = False

    def __post_init__(self):
        """Validate and process configuration."""
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created cache directory: {self.cache_dir}")


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders with professional features."""

    def __init__(self, config: Optional[LoaderConfig] = None):
        """Initialize loader with configuration.

        Args:
            config: Loader configuration
        """
        self.config = config or LoaderConfig()
        self._stats = {
            "loads_successful": 0,
            "loads_failed": 0,
            "total_samples": 0,
            "cache_hits": 0,
            "validation_failures": 0,
        }
        self._cache = {}

    @abstractmethod
    def load(self, source: Union[str, Path], **kwargs) -> Any:
        """Load dataset from source.

        Args:
            source: Dataset source (path, URL, or identifier)
            **kwargs: Additional loading parameters

        Returns:
            Loaded dataset
        """
        pass

    @abstractmethod
    def validate(self, dataset: Any) -> bool:
        """Validate loaded dataset.

        Args:
            dataset: Dataset to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails and strict mode
        """
        pass

    def _cache_key(self, source: Union[str, Path], **kwargs) -> str:
        """Generate cache key for dataset.

        Args:
            source: Dataset source
            **kwargs: Loading parameters

        Returns:
            Cache key
        """
        key_parts = [str(source)]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get dataset from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached dataset or None
        """
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            logger.debug(f"Cache hit: {cache_key}")
            return self._cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, dataset: Any):
        """Add dataset to cache.

        Args:
            cache_key: Cache key
            dataset: Dataset to cache
        """
        self._cache[cache_key] = dataset
        logger.debug(f"Cached dataset: {cache_key}")

    def load_with_retry(self, source: Union[str, Path], **kwargs) -> Any:
        """Load dataset with retry logic.

        Args:
            source: Dataset source
            **kwargs: Loading parameters

        Returns:
            Loaded dataset

        Raises:
            RuntimeError: After all retries exhausted
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Check cache first
                cache_key = self._cache_key(source, **kwargs)
                cached = self._get_from_cache(cache_key)
                if cached is not None:
                    return cached

                # Load dataset
                dataset = self.load(source, **kwargs)

                # Validate if configured
                if self.config.validate_on_load:
                    if not self.validate(dataset):
                        self._stats["validation_failures"] += 1
                        raise ValueError(f"Dataset validation failed for {source}")

                # Cache and return
                self._add_to_cache(cache_key, dataset)
                self._stats["loads_successful"] += 1

                # Update sample count
                if hasattr(dataset, "__len__"):
                    self._stats["total_samples"] += len(dataset)

                return dataset

            except Exception as e:
                last_error = e
                self._stats["loads_failed"] += 1

                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"Load attempt {attempt + 1} failed: {e}. Retrying..."
                    )
                    if self.config.retry_delay > 0:
                        import time

                        time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All load attempts failed for {source}")

        raise RuntimeError(
            f"Failed to load dataset after {self.config.max_retries} attempts: {last_error}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics.

        Returns:
            Statistics dictionary
        """
        return self._stats.copy()


class HuggingFaceLoader(DatasetLoader):
    """Loader for HuggingFace datasets with advanced features."""

    def __init__(self, config: Optional[LoaderConfig] = None):
        """Initialize HuggingFace loader.

        Args:
            config: Loader configuration
        """
        super().__init__(config)
        self._datasets_available = None

    @property
    def datasets_available(self) -> bool:
        """Check if datasets library is available."""
        if self._datasets_available is None:
            try:
                import datasets  # noqa: F401

                self._datasets_available = True
            except ImportError:
                self._datasets_available = False
        return self._datasets_available

    def load(self, source: str, **kwargs) -> Any:
        """Load dataset from HuggingFace with robust error handling.

        Args:
            source: Dataset name on HuggingFace
            **kwargs: Additional parameters (split, config, etc.)

        Returns:
            HuggingFace dataset object

        Raises:
            ImportError: If datasets library not available
            RuntimeError: If loading fails
        """
        if not self.datasets_available:
            raise ImportError(
                "datasets library not installed. Run: pip install datasets"
            )

        try:
            from datasets import load_dataset

            # Extract and process parameters
            split = kwargs.pop("split", None)
            config_name = kwargs.pop("config", None)
            trust_remote = kwargs.pop(
                "trust_remote_code", self.config.trust_remote_code
            )
            streaming = kwargs.pop("streaming", self.config.streaming)

            # Prepare cache directory
            cache_dir = None
            if self.config.cache_dir:
                cache_dir = str(self.config.cache_dir / "huggingface")
                Path(cache_dir).mkdir(parents=True, exist_ok=True)

            # Load dataset
            logger.info(
                f"Loading HuggingFace dataset: {source} (split={split}, config={config_name})"
            )

            dataset = load_dataset(
                source,
                config_name,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote,
                streaming=streaming,
                **kwargs,
            )

            logger.info(f"Successfully loaded dataset: {source}")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {source}: {e}")
            raise RuntimeError(f"HuggingFace dataset loading failed: {e}") from e

    def validate(self, dataset: Any) -> bool:
        """Validate HuggingFace dataset with comprehensive checks.

        Args:
            dataset: Dataset to validate

        Returns:
            True if valid
        """
        try:
            # Check for Dataset attributes
            if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
                # Check if empty when not allowed
                if not self.config.allow_empty and len(dataset) == 0:
                    logger.error("Dataset is empty and allow_empty=False")
                    return False
                return True

            # Check for DatasetDict
            if hasattr(dataset, "keys") and callable(dataset.keys):
                # Validate each split
                for split_name in dataset.keys():
                    split_data = dataset[split_name]
                    if not self.config.allow_empty and len(split_data) == 0:
                        logger.error(
                            f"Split '{split_name}' is empty and allow_empty=False"
                        )
                        return False
                return True

            # Check for IterableDataset (streaming)
            if hasattr(dataset, "__iter__"):
                logger.warning("Streaming dataset detected - cannot validate size")
                return True

            return False

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False


class JSONLoader(DatasetLoader):
    """Loader for JSON datasets with flexible format handling."""

    def __init__(self, config: Optional[LoaderConfig] = None):
        """Initialize JSON loader.

        Args:
            config: Loader configuration
        """
        super().__init__(config)
        self.supported_encodings = ["utf-8", "utf-8-sig", "latin-1", "ascii"]

    def load(self, source: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
        """Load dataset from JSON file with encoding detection.

        Args:
            source: Path to JSON file
            **kwargs: Additional parameters (encoding, data_key)

        Returns:
            List of dictionaries

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        encoding = kwargs.get("encoding", "utf-8")
        data_key = kwargs.get("data_key", None)

        logger.info(f"Loading JSON dataset: {path}")

        # Try loading with different encodings if needed
        last_error = None
        for enc in [encoding] + [e for e in self.supported_encodings if e != encoding]:
            try:
                with open(path, "r", encoding=enc) as f:
                    data = json.load(f)
                logger.debug(f"Successfully loaded with encoding: {enc}")
                break
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                last_error = e
                continue
        else:
            raise ValueError(f"Failed to load JSON file: {last_error}")

        # Handle different JSON structures
        if isinstance(data, dict):
            if data_key and data_key in data:
                data = data[data_key]
            elif "data" in data:
                data = data["data"]
            elif "samples" in data:
                data = data["samples"]
            elif "items" in data:
                data = data["items"]
            elif "records" in data:
                data = data["records"]
            else:
                # Convert single dict to list
                data = [data]

        if not isinstance(data, list):
            raise ValueError(f"Expected list or dict with data key, got {type(data)}")

        logger.info(f"Loaded {len(data)} samples from JSON")
        return data

    def validate(self, dataset: Any) -> bool:
        """Validate JSON dataset structure.

        Args:
            dataset: Dataset to validate

        Returns:
            True if valid
        """
        if not isinstance(dataset, list):
            logger.error(f"Dataset must be a list, got {type(dataset)}")
            return False

        if not self.config.allow_empty and len(dataset) == 0:
            logger.error("Dataset is empty and allow_empty=False")
            return False

        # Check if all items are dictionaries
        non_dict_items = [
            i for i, item in enumerate(dataset) if not isinstance(item, dict)
        ]
        if non_dict_items:
            logger.error(f"Non-dict items at indices: {non_dict_items[:5]}...")
            return False

        return True


class CSVLoader(DatasetLoader):
    """Loader for CSV datasets with pandas integration."""

    def __init__(self, config: Optional[LoaderConfig] = None):
        """Initialize CSV loader.

        Args:
            config: Loader configuration
        """
        super().__init__(config)

    def load(self, source: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load dataset from CSV file with robust handling.

        Args:
            source: Path to CSV file
            **kwargs: Additional pandas parameters

        Returns:
            Pandas DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is invalid
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        logger.info(f"Loading CSV dataset: {path}")

        try:
            # Load with pandas
            df = pd.read_csv(path, **kwargs)

            # Handle empty DataFrames
            if df.empty and not self.config.allow_empty:
                raise ValueError("CSV file resulted in empty DataFrame")

            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from CSV")
            return df

        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise ValueError(f"CSV loading failed: {e}") from e

    def validate(self, dataset: Any) -> bool:
        """Validate CSV dataset.

        Args:
            dataset: Dataset to validate

        Returns:
            True if valid
        """
        if not isinstance(dataset, pd.DataFrame):
            logger.error(f"Dataset must be a DataFrame, got {type(dataset)}")
            return False

        if not self.config.allow_empty and dataset.empty:
            logger.error("DataFrame is empty and allow_empty=False")
            return False

        # Check for NaN-only columns
        nan_cols = dataset.columns[dataset.isna().all()].tolist()
        if nan_cols:
            logger.warning(f"Columns with only NaN values: {nan_cols}")

        return True


class ParquetLoader(DatasetLoader):
    """Loader for Parquet datasets."""

    def load(self, source: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load dataset from Parquet file.

        Args:
            source: Path to Parquet file
            **kwargs: Additional parameters

        Returns:
            Pandas DataFrame
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            raise ImportError("pyarrow required for Parquet support")

        logger.info(f"Loading Parquet dataset: {path}")

        df = pd.read_parquet(path, **kwargs)
        logger.info(f"Loaded {len(df)} rows from Parquet")

        return df

    def validate(self, dataset: Any) -> bool:
        """Validate Parquet dataset."""
        return isinstance(dataset, pd.DataFrame) and (
            self.config.allow_empty or not dataset.empty
        )


class DatasetLoaderFactory:
    """Factory for creating dataset loaders with registration support."""

    _loaders: Dict[str, Type[DatasetLoader]] = {
        "huggingface": HuggingFaceLoader,
        "hf": HuggingFaceLoader,  # Alias
        "json": JSONLoader,
        "jsonl": JSONLoader,  # Will handle in load_dataset
        "csv": CSVLoader,
        "parquet": ParquetLoader,
    }

    _instance = None

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_loader(
        cls, loader_type: str, config: Optional[LoaderConfig] = None
    ) -> DatasetLoader:
        """Get dataset loader by type.

        Args:
            loader_type: Type of loader
            config: Loader configuration

        Returns:
            DatasetLoader instance

        Raises:
            ValueError: If loader type not found
        """
        loader_type = loader_type.lower()

        if loader_type not in cls._loaders:
            available = ", ".join(sorted(cls._loaders.keys()))
            raise ValueError(
                f"Unknown loader type: {loader_type}. Available: {available}"
            )

        loader_class = cls._loaders[loader_type]
        return loader_class(config)

    @classmethod
    def register_loader(
        cls, name: str, loader_class: Type[DatasetLoader], override: bool = False
    ):
        """Register a custom loader.

        Args:
            name: Loader name
            loader_class: Loader class (must inherit from DatasetLoader)
            override: Whether to override existing loader

        Raises:
            TypeError: If loader_class doesn't inherit from DatasetLoader
            ValueError: If name exists and override=False
        """
        if not issubclass(loader_class, DatasetLoader):
            raise TypeError(f"{loader_class} must inherit from DatasetLoader")

        name = name.lower()

        if name in cls._loaders and not override:
            raise ValueError(
                f"Loader '{name}' already registered. Use override=True to replace."
            )

        cls._loaders[name] = loader_class
        logger.info(f"Registered loader: {name} -> {loader_class.__name__}")

    @classmethod
    def list_loaders(cls) -> List[str]:
        """List available loader types.

        Returns:
            List of loader names
        """
        return sorted(cls._loaders.keys())


@lru_cache(maxsize=32)
def _detect_loader_type(source_str: str) -> str:
    """Detect loader type from source string.

    Args:
        source_str: Source string

    Returns:
        Detected loader type
    """
    source_lower = source_str.lower()
    path = Path(source_str)

    # Check file extensions
    if source_lower.endswith(".json"):
        return "json"
    elif source_lower.endswith(".jsonl"):
        return "jsonl"
    elif source_lower.endswith(".csv"):
        return "csv"
    elif source_lower.endswith(".parquet") or source_lower.endswith(".pq"):
        return "parquet"
    elif path.exists():
        # File exists but unknown extension
        return "json"  # Default to JSON
    elif "/" in source_str and not path.exists():
        # Likely a HuggingFace dataset
        return "huggingface"
    else:
        # Default to HuggingFace
        return "huggingface"


def loader_factory(
    source: Optional[Union[str, Path]] = None,
    loader_type: str = "auto",
    config: Optional[LoaderConfig] = None,
) -> DatasetLoader:
    """Factory function to create dataset loaders.

    Args:
        source: Optional dataset source for auto-detection
        loader_type: Loader type or "auto" for detection
        config: Loader configuration

    Returns:
        Dataset loader instance
    """
    # Auto-detect loader type if needed
    if loader_type == "auto" and source:
        loader_type = _detect_loader_type(str(source))
        logger.debug(f"Auto-detected loader type: {loader_type}")

    # Get appropriate loader
    factory = DatasetLoaderFactory()
    return factory.get_loader(loader_type, config or LoaderConfig())


def load_dataset(
    source: Union[str, Path],
    loader_type: str = "auto",
    config: Optional[LoaderConfig] = None,
    **kwargs,
) -> Any:
    """Load dataset with automatic loader detection and robust error handling.

    Args:
        source: Dataset source
        loader_type: Loader type or "auto" for detection
        config: Loader configuration
        **kwargs: Additional parameters for specific loader

    Returns:
        Loaded dataset

    Raises:
        ValueError: If loading or validation fails
    """
    # Auto-detect loader type if needed
    if loader_type == "auto":
        loader_type = _detect_loader_type(str(source))
        logger.debug(f"Auto-detected loader type: {loader_type}")

    # Handle JSONL specifically
    if loader_type == "jsonl":
        loader_type = "json"
        kwargs["jsonl"] = True

    # Get appropriate loader
    factory = DatasetLoaderFactory()
    loader = factory.get_loader(loader_type, config)

    # Load with retry logic
    dataset = loader.load_with_retry(source, **kwargs)

    logger.info(f"Dataset loaded successfully with {loader_type} loader")
    return dataset


# Module exports
__all__ = [
    "DatasetLoader",
    "LoaderConfig",
    "loader_factory",
    "HuggingFaceLoader",
    "JSONLoader",
    "CSVLoader",
    "ParquetLoader",
    "DatasetLoaderFactory",
    "load_dataset",
]
