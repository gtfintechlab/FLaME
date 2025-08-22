"""Fixed LLM client with true parallel batch processing."""

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench_forge.llm.config import LLMConfig

logger = logging.getLogger(__name__)


class ResponseCache:
    """Simple file-based response cache."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache."""
        self.cache_dir = cache_dir or Path(".cache/llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model."""
        cache_input = f"{prompt}:{model}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response."""
        key = self._get_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.txt"

        if cache_file.exists():
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            return cache_file.read_text()

        return None

    def set(self, prompt: str, model: str, response: str):
        """Set cached response."""
        key = self._get_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.txt"
        cache_file.write_text(response)
        logger.debug(f"Cached response for prompt: {prompt[:50]}...")


@dataclass
class LLMClient:
    """LLM client with proper parallel batch processing using litellm."""

    config: LLMConfig
    cache: Optional[ResponseCache] = None
    stats: Dict[str, Any] = field(
        default_factory=lambda: {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "total_time_seconds": 0.0,
        }
    )

    def __post_init__(self):
        """Initialize LLM client."""
        # Setup caching if enabled
        if self.config.cache_responses:
            cache_dir = Path(".cache") / "llm_responses"
            self.cache = ResponseCache(cache_dir)
            logger.info(f"Response caching enabled at {cache_dir}")

        # Import litellm for batch processing
        try:
            import litellm

            self.litellm = litellm
            # Configure litellm
            litellm.set_verbose = False
            if hasattr(self.config, "base_url") and self.config.base_url:
                litellm.api_base = self.config.base_url
            logger.info(
                "Initialized LLMClient with litellm for parallel batch processing"
            )
        except ImportError:
            logger.error("litellm not installed. Install with: pip install litellm")
            raise

    def complete(self, prompt: str, **kwargs) -> str:
        """Complete a single prompt (fallback for single requests)."""
        self.stats["total_requests"] += 1

        # Check cache
        if self.cache:
            cached = self.cache.get(prompt, self.config.model)
            if cached:
                self.stats["cache_hits"] += 1
                return cached
            self.stats["cache_misses"] += 1

        # Make API call using litellm
        start_time = datetime.now()

        try:
            # Prepare kwargs
            completion_kwargs = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }

            if self.config.top_k:
                completion_kwargs["top_k"] = self.config.top_k

            # Call litellm
            response = self.litellm.completion(**completion_kwargs)
            result = response.choices[0].message.content

            # Update stats
            elapsed = (datetime.now() - start_time).total_seconds()
            self.stats["total_time_seconds"] += elapsed
            if hasattr(response, "usage"):
                self.stats["total_tokens"] += response.usage.total_tokens

            # Cache response
            if self.cache:
                self.cache.set(prompt, self.config.model, result)

            return result

        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise

    def complete_batch(
        self, prompts: List[str], show_progress: bool = True, **kwargs
    ) -> List[str]:
        """Complete multiple prompts in PARALLEL using litellm.batch_completion.

        This is the key fix for performance - use true parallel processing!

        Args:
            prompts: List of prompts to complete
            show_progress: Whether to show progress bar
            **kwargs: Additional completion arguments

        Returns:
            List of responses in the same order as prompts
        """
        if not prompts:
            return []

        logger.info(f"Processing batch of {len(prompts)} prompts in PARALLEL")
        self.stats["total_requests"] += len(prompts)

        # Check cache for all prompts
        responses = [None] * len(prompts)
        uncached_indices = []
        uncached_prompts = []

        if self.cache:
            for i, prompt in enumerate(prompts):
                cached = self.cache.get(prompt, self.config.model)
                if cached:
                    responses[i] = cached
                    self.stats["cache_hits"] += 1
                else:
                    uncached_indices.append(i)
                    uncached_prompts.append(prompt)
                    self.stats["cache_misses"] += 1
        else:
            uncached_indices = list(range(len(prompts)))
            uncached_prompts = prompts

        # Process uncached prompts in parallel if any
        if uncached_prompts:
            logger.info(
                f"Making parallel API calls for {len(uncached_prompts)} uncached prompts"
            )
            start_time = datetime.now()

            try:
                # Prepare messages for batch_completion
                messages_batch = [
                    [{"role": "user", "content": prompt}] for prompt in uncached_prompts
                ]

                # Build parameters for batch_completion
                batch_params = {
                    "model": self.config.model,
                    "messages": messages_batch,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "top_p": kwargs.get("top_p", self.config.top_p),
                    "num_retries": 3,  # Built-in retry mechanism
                }

                if self.config.top_k:
                    batch_params["top_k"] = self.config.top_k

                # Use litellm's batch_completion for TRUE PARALLEL PROCESSING
                batch_responses = self.litellm.batch_completion(**batch_params)

                # Extract responses and update cache
                for idx, response in enumerate(batch_responses):
                    original_idx = uncached_indices[idx]
                    if hasattr(response, "choices") and response.choices:
                        result = response.choices[0].message.content
                    else:
                        # Handle the response object directly
                        result = response

                    responses[original_idx] = result

                    # Cache the response
                    if self.cache:
                        self.cache.set(uncached_prompts[idx], self.config.model, result)

                    # Update token stats
                    if hasattr(response, "usage"):
                        self.stats["total_tokens"] += response.usage.total_tokens

                # Update timing stats
                elapsed = (datetime.now() - start_time).total_seconds()
                self.stats["total_time_seconds"] += elapsed
                logger.info(
                    f"Parallel batch processing completed in {elapsed:.2f} seconds"
                )

            except Exception as e:
                logger.error(f"Batch completion failed: {e}")
                raise

        # Convert response objects to strings if needed
        final_responses = []
        for response in responses:
            if hasattr(response, "choices"):
                # It's a response object, extract the content
                final_responses.append(response.choices[0].message.content)
            else:
                # It's already a string
                final_responses.append(response)

        return final_responses

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self.stats.copy()

        # Calculate cache hit rate
        if stats["total_requests"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_requests"]
            stats["avg_time_per_request"] = (
                stats["total_time_seconds"] / stats["total_requests"]
            )
        else:
            stats["cache_hit_rate"] = 0.0
            stats["avg_time_per_request"] = 0.0

        return stats

    def clear_cache(self):
        """Clear response cache."""
        if self.cache and self.cache.cache_dir.exists():
            for file in self.cache.cache_dir.glob("*.txt"):
                file.unlink()
            logger.info("Cache cleared")

    def __repr__(self) -> str:
        """String representation."""
        return f"LLMClient(model={self.config.model}, provider={self.config.provider}, parallel=True)"
