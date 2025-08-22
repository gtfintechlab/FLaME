"""Caching utilities for BenchForge.

Professional-grade caching system with namespace support,
expiration, and multiple backend support.
"""

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
from functools import wraps
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache management."""

    cache_dir: Optional[Path] = None
    default_ttl: int = 86400  # 24 hours in seconds
    max_size_mb: float = 1000  # Maximum cache size in MB
    enable_stats: bool = True
    compression: bool = False
    auto_cleanup: bool = True
    cleanup_interval: int = 3600  # 1 hour

    def __post_init__(self):
        """Process and validate configuration."""
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "benchforge"
        else:
            self.cache_dir = Path(self.cache_dir)

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.default_ttl < 0:
            raise ValueError(
                f"default_ttl must be non-negative, got {self.default_ttl}"
            )

        if self.max_size_mb <= 0:
            raise ValueError(f"max_size_mb must be positive, got {self.max_size_mb}")


class CacheManager:
    """Manage dataset and response caching with professional features."""

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager.

        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._lock = threading.Lock()
        self._metadata_file = self.config.cache_dir / "metadata.json"
        self._metadata = self._load_metadata()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "bytes_saved": 0,
            "bytes_loaded": 0,
            "errors": 0,
        }

        # Start cleanup thread if configured
        if self.config.auto_cleanup:
            self._start_cleanup_thread()

        logger.info(f"CacheManager initialized with cache_dir: {self.config.cache_dir}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata.

        Returns:
            Metadata dictionary
        """
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

        return {"entries": {}, "namespaces": {}}

    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _get_cache_key(self, key: str, namespace: str = "default") -> str:
        """Generate cache key.

        Args:
            key: Original key
            namespace: Cache namespace

        Returns:
            Hash-based cache key
        """
        full_key = f"{namespace}:{key}"
        hash_key = hashlib.sha256(full_key.encode()).hexdigest()
        return hash_key

    def _get_cache_path(self, cache_key: str, extension: str = ".pkl") -> Path:
        """Get cache file path.

        Args:
            cache_key: Cache key
            extension: File extension

        Returns:
            Path to cache file
        """
        # Use subdirectories to avoid too many files in one directory
        subdir = cache_key[:2]
        cache_subdir = self.config.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}{extension}"

    def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if cache exists and is valid.

        Args:
            key: Cache key
            namespace: Cache namespace

        Returns:
            True if cached and valid
        """
        cache_key = self._get_cache_key(key, namespace)

        # Check metadata
        if cache_key in self._metadata.get("entries", {}):
            entry = self._metadata["entries"][cache_key]

            # Check expiration
            if "expires" in entry:
                expires = datetime.fromisoformat(entry["expires"])
                if expires < datetime.now():
                    # Expired
                    self.delete(key, namespace)
                    return False

            # Check file exists
            cache_path = self._get_cache_path(cache_key)
            return cache_path.exists()

        return False

    def get(
        self, key: str, namespace: str = "default", default: Any = None
    ) -> Optional[Any]:
        """Get cached value with expiration check.

        Args:
            key: Cache key
            namespace: Cache namespace
            default: Default value if not cached

        Returns:
            Cached value or default
        """
        cache_key = self._get_cache_key(key, namespace)
        cache_path = self._get_cache_path(cache_key)

        with self._lock:
            # Check if exists and valid
            if not self.exists(key, namespace):
                self._stats["misses"] += 1
                return default

            try:
                # Load from cache
                if self.config.compression:
                    import gzip

                    with gzip.open(cache_path, "rb") as f:
                        value = pickle.load(f)
                else:
                    with open(cache_path, "rb") as f:
                        value = pickle.load(f)

                # Update statistics
                self._stats["hits"] += 1
                self._stats["bytes_loaded"] += cache_path.stat().st_size

                # Update access time
                if cache_key in self._metadata.get("entries", {}):
                    self._metadata["entries"][cache_key]["last_access"] = (
                        datetime.now().isoformat()
                    )
                    self._save_metadata()

                logger.debug(f"Cache hit: {key} (namespace: {namespace})")
                return value

            except Exception as e:
                logger.error(f"Failed to load cache {key}: {e}")
                self._stats["errors"] += 1
                return default

    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[int] = None,
    ) -> bool:
        """Set cache value with TTL.

        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl: Time to live in seconds (None for no expiration)

        Returns:
            True if successful
        """
        cache_key = self._get_cache_key(key, namespace)
        cache_path = self._get_cache_path(cache_key)

        with self._lock:
            try:
                # Check cache size limit
                if self.config.auto_cleanup:
                    self._check_size_limit()

                # Save to cache
                if self.config.compression:
                    import gzip

                    with gzip.open(cache_path, "wb") as f:
                        pickle.dump(value, f)
                else:
                    with open(cache_path, "wb") as f:
                        pickle.dump(value, f)

                # Update metadata
                ttl = ttl if ttl is not None else self.config.default_ttl
                entry = {
                    "key": key,
                    "namespace": namespace,
                    "created": datetime.now().isoformat(),
                    "last_access": datetime.now().isoformat(),
                    "size": cache_path.stat().st_size,
                }

                if ttl > 0:
                    entry["expires"] = (
                        datetime.now() + timedelta(seconds=ttl)
                    ).isoformat()

                if "entries" not in self._metadata:
                    self._metadata["entries"] = {}
                self._metadata["entries"][cache_key] = entry

                # Update namespace tracking
                if "namespaces" not in self._metadata:
                    self._metadata["namespaces"] = {}
                if namespace not in self._metadata["namespaces"]:
                    self._metadata["namespaces"][namespace] = []
                if cache_key not in self._metadata["namespaces"][namespace]:
                    self._metadata["namespaces"][namespace].append(cache_key)

                self._save_metadata()

                # Update statistics
                self._stats["bytes_saved"] += entry["size"]

                logger.debug(f"Cached: {key} (namespace: {namespace}, ttl: {ttl}s)")
                return True

            except Exception as e:
                logger.error(f"Failed to cache {key}: {e}")
                self._stats["errors"] += 1
                return False

    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete cached value.

        Args:
            key: Cache key
            namespace: Cache namespace

        Returns:
            True if deleted
        """
        cache_key = self._get_cache_key(key, namespace)
        cache_path = self._get_cache_path(cache_key)

        with self._lock:
            # Remove file
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete cache file: {e}")
                    return False

            # Remove from metadata
            if cache_key in self._metadata.get("entries", {}):
                del self._metadata["entries"][cache_key]

            if namespace in self._metadata.get("namespaces", {}):
                if cache_key in self._metadata["namespaces"][namespace]:
                    self._metadata["namespaces"][namespace].remove(cache_key)

            self._save_metadata()

            logger.debug(f"Deleted cache: {key} (namespace: {namespace})")
            return True

    def clear(self, namespace: Optional[str] = None):
        """Clear cache.

        Args:
            namespace: Clear specific namespace or all if None
        """
        with self._lock:
            if namespace:
                # Clear specific namespace
                if namespace in self._metadata.get("namespaces", {}):
                    keys_to_delete = self._metadata["namespaces"][namespace].copy()
                    count = 0

                    for cache_key in keys_to_delete:
                        cache_path = self._get_cache_path(cache_key)
                        if cache_path.exists():
                            try:
                                cache_path.unlink()
                                count += 1
                            except Exception as e:
                                logger.error(f"Failed to delete {cache_path}: {e}")

                        if cache_key in self._metadata.get("entries", {}):
                            del self._metadata["entries"][cache_key]

                    del self._metadata["namespaces"][namespace]
                    self._save_metadata()

                    logger.info(f"Cleared {count} items from namespace: {namespace}")
            else:
                # Clear all
                count = 0
                for subdir in self.config.cache_dir.iterdir():
                    if (
                        subdir.is_dir() and len(subdir.name) == 2
                    ):  # Cache subdirectories
                        for cache_file in subdir.glob("*.pkl*"):
                            try:
                                cache_file.unlink()
                                count += 1
                            except Exception as e:
                                logger.error(f"Failed to delete {cache_file}: {e}")

                # Clear metadata
                self._metadata = {"entries": {}, "namespaces": {}}
                self._save_metadata()

                logger.info(f"Cleared {count} cached items")

    def _check_size_limit(self):
        """Check and enforce cache size limit."""
        total_size = 0
        entries_by_access = []

        # Calculate total size and sort by access time
        for cache_key, entry in self._metadata.get("entries", {}).items():
            total_size += entry.get("size", 0)
            last_access = entry.get("last_access", entry.get("created"))
            entries_by_access.append((last_access, cache_key))

        max_size_bytes = self.config.max_size_mb * 1024 * 1024

        if total_size > max_size_bytes:
            # Sort by last access (oldest first)
            entries_by_access.sort()

            # Evict oldest entries until under limit
            for last_access, cache_key in entries_by_access:
                if total_size <= max_size_bytes * 0.9:  # Leave 10% buffer
                    break

                # Get entry info
                entry = self._metadata["entries"].get(cache_key, {})
                namespace = entry.get("namespace", "default")
                key = entry.get("key", "unknown")

                # Delete entry
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    try:
                        size = cache_path.stat().st_size
                        cache_path.unlink()
                        total_size -= size
                        self._stats["evictions"] += 1
                        logger.debug(f"Evicted cache: {key} (namespace: {namespace})")
                    except Exception as e:
                        logger.error(f"Failed to evict {cache_path}: {e}")

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        import threading

        def cleanup_worker():
            while True:
                time.sleep(self.config.cleanup_interval)
                try:
                    self._cleanup_expired()
                    self._check_size_limit()
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")

        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
        logger.debug("Started cache cleanup thread")

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        now = datetime.now()
        expired_keys = []

        for cache_key, entry in self._metadata.get("entries", {}).items():
            if "expires" in entry:
                expires = datetime.fromisoformat(entry["expires"])
                if expires < now:
                    expired_keys.append(cache_key)

        for cache_key in expired_keys:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete expired cache: {e}")

            if cache_key in self._metadata["entries"]:
                del self._metadata["entries"][cache_key]

        if expired_keys:
            self._save_metadata()
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def cached(
        self,
        namespace: str = "default",
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None,
    ):
        """Decorator for caching function results.

        Args:
            namespace: Cache namespace
            ttl: Time to live in seconds
            key_func: Custom key generation function

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    key = "|".join(key_parts)

                # Check cache
                cached_value = self.get(key, namespace)
                if cached_value is not None:
                    return cached_value

                # Compute and cache
                result = func(*args, **kwargs)
                self.set(key, result, namespace, ttl)
                return result

            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper

        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Statistics dictionary
        """
        stats = self._stats.copy()

        # Add computed statistics
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = stats["hits"] / total_requests
        else:
            stats["hit_rate"] = 0.0

        # Add cache size info
        total_size = sum(
            entry.get("size", 0) for entry in self._metadata.get("entries", {}).values()
        )
        stats["total_size_mb"] = total_size / (1024 * 1024)
        stats["num_entries"] = len(self._metadata.get("entries", {}))
        stats["num_namespaces"] = len(self._metadata.get("namespaces", {}))

        return stats


class ResponseCache:
    """Specialized cache for LLM responses."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize response cache.

        Args:
            cache_manager: Cache manager instance
        """
        self.cache_manager = cache_manager or CacheManager()
        self.namespace = "llm_responses"
        self._stats = {"cache_hits": 0, "cache_misses": 0, "responses_cached": 0}

    def get_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Get cached response.

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Generation temperature
            max_tokens: Maximum tokens

        Returns:
            Cached response or None
        """
        # Generate cache key including parameters
        key_parts = [
            f"model:{model}",
            f"temp:{temperature}",
            f"max_tokens:{max_tokens}",
            f"prompt_hash:{hashlib.md5(prompt.encode()).hexdigest()}",
        ]
        key = "|".join(key_parts)

        response = self.cache_manager.get(key, self.namespace)

        if response is not None:
            self._stats["cache_hits"] += 1
            logger.debug(f"Response cache hit for model {model}")
        else:
            self._stats["cache_misses"] += 1

        return response

    def cache_response(
        self,
        prompt: str,
        model: str,
        response: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        ttl: Optional[int] = None,
    ):
        """Cache a response.

        Args:
            prompt: Input prompt
            model: Model name
            response: Model response
            temperature: Generation temperature
            max_tokens: Maximum tokens
            ttl: Time to live in seconds
        """
        # Generate cache key including parameters
        key_parts = [
            f"model:{model}",
            f"temp:{temperature}",
            f"max_tokens:{max_tokens}",
            f"prompt_hash:{hashlib.md5(prompt.encode()).hexdigest()}",
        ]
        key = "|".join(key_parts)

        success = self.cache_manager.set(key, response, self.namespace, ttl)

        if success:
            self._stats["responses_cached"] += 1
            logger.debug(f"Cached response for model {model}")

    def clear_model_cache(self, model: str):
        """Clear cache for specific model.

        Args:
            model: Model name
        """
        # Get all entries in namespace
        entries_to_delete = []

        for cache_key, entry in self.cache_manager._metadata.get("entries", {}).items():
            if entry.get("namespace") == self.namespace:
                key = entry.get("key", "")
                if f"model:{model}" in key:
                    entries_to_delete.append((cache_key, key))

        # Delete matching entries
        count = 0
        for cache_key, key in entries_to_delete:
            # Extract original key from metadata key
            if self.cache_manager.delete(key, self.namespace):
                count += 1

        logger.info(f"Cleared {count} cached responses for model: {model}")

    def get_stats(self) -> Dict[str, Any]:
        """Get response cache statistics.

        Returns:
            Statistics dictionary
        """
        stats = self._stats.copy()

        # Add hit rate
        total = stats["cache_hits"] + stats["cache_misses"]
        if total > 0:
            stats["hit_rate"] = stats["cache_hits"] / total
        else:
            stats["hit_rate"] = 0.0

        return stats


# Module exports
__all__ = [
    "CacheConfig",
    "CacheManager",
    "ResponseCache",
]
