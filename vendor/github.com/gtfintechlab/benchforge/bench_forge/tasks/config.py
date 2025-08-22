"""Task configuration with validation and type safety."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pathlib import Path


class PromptFormat(Enum):
    """Supported prompt formats."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    INSTRUCTION = "instruction"
    CUSTOM = "custom"


@dataclass
class TaskConfig:
    """Configuration for a benchmark task with validation.

    Attributes:
        name: Unique task identifier
        dataset: Dataset name or path
        dataset_split: Split to use (train/validation/test)
        metrics: List of metric names to compute
        prompt_format: Default prompt format
        batch_size: Processing batch size
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        seed: Random seed for reproducibility
        num_samples: Number of samples to process (None = all)
        cache_dir: Directory for caching
        output_dir: Directory for outputs
        metadata: Additional task-specific configuration
    """

    name: str
    dataset: str = ""
    dataset_split: str = "test"
    metrics: List[str] = field(default_factory=list)
    prompt_format: Union[str, PromptFormat] = PromptFormat.ZERO_SHOT
    batch_size: int = 10
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    seed: int = 42
    num_samples: Optional[int] = None
    cache_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Convert string to enum if needed
        if isinstance(self.prompt_format, str):
            try:
                self.prompt_format = PromptFormat(self.prompt_format)
            except ValueError:
                raise ValueError(f"Invalid prompt format: {self.prompt_format}")

        # Validate ranges
        if not 0 <= self.temperature <= 2:
            raise ValueError(
                f"Temperature must be between 0 and 2, got {self.temperature}"
            )
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {self.top_p}")
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.max_tokens < 1:
            raise ValueError(f"Max tokens must be positive, got {self.max_tokens}")

        # Convert paths
        if self.cache_dir and not isinstance(self.cache_dir, Path):
            self.cache_dir = Path(self.cache_dir)
        if self.output_dir and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, PromptFormat):
                result[key] = value.value
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
