"""LLM configuration with validation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for LLM client.

    Attributes:
        provider: Provider name (litellm, openai, anthropic, etc.)
        model: Model identifier
        api_key: API key (if not in environment)
        base_url: Base URL for API
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        seed: Random seed for reproducibility
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay in seconds
        cache_responses: Whether to cache responses
        cache_dir: Directory for response cache
        extra_params: Additional provider-specific parameters
    """

    provider: str = "litellm"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    seed: Optional[int] = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_responses: bool = True
    cache_dir: Optional[Path] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.temperature <= 2:
            raise ValueError(
                f"Temperature must be between 0 and 2, got {self.temperature}"
            )
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {self.top_p}")
        if self.max_tokens < 1:
            raise ValueError(f"Max tokens must be positive, got {self.max_tokens}")
        if self.timeout < 1:
            raise ValueError(f"Timeout must be positive, got {self.timeout}")
        if self.max_retries < 0:
            raise ValueError(
                f"Max retries must be non-negative, got {self.max_retries}"
            )

        if self.cache_dir and not isinstance(self.cache_dir, Path):
            self.cache_dir = Path(self.cache_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if self.top_k is not None:
            result["top_k"] = self.top_k
        if self.seed is not None:
            result["seed"] = self.seed
        if self.api_key:
            result["api_key"] = self.api_key
        if self.base_url:
            result["base_url"] = self.base_url

        # Add extra parameters
        result.update(self.extra_params)

        return result
