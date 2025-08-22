"""Batch processing with retry logic and parallelization."""

import time
import logging
from typing import Any, Callable, List, Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    jitter: bool = True
    max_workers: int = 4
    timeout: Optional[float] = None


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split a list into chunks of specified size.

    Args:
        items: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    if chunk_size < 1:
        raise ValueError(f"Chunk size must be positive, got {chunk_size}")

    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i : i + chunk_size])

    return chunks


class BatchProcessor:
    """Process items in batches with retry logic and optional parallelization."""

    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()

        # Statistics
        self.stats = {
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_retries": 0,
            "total_items": 0,
        }

        logger.info(
            f"Initialized BatchProcessor with batch_size={self.config.batch_size}"
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        if self.config.exponential_backoff:
            delay = self.config.retry_delay * (2**attempt)
        else:
            delay = self.config.retry_delay

        if self.config.jitter:
            # Add random jitter (±25%)
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay += jitter

        return max(0, delay)

    def _process_batch_with_retry(
        self, batch: List[T], process_fn: Callable[[List[T]], List[Any]], batch_idx: int
    ) -> List[Any]:
        """Process a single batch with retry logic.

        Args:
            batch: Items to process
            process_fn: Function to process the batch
            batch_idx: Batch index for logging

        Returns:
            Processed results

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Process batch
                results = process_fn(batch)

                # Validate results
                if len(results) != len(batch):
                    raise ValueError(
                        f"Process function returned {len(results)} results for {len(batch)} items"
                    )

                logger.debug(f"Batch {batch_idx} processed successfully")
                return results

            except Exception as e:
                last_error = e
                self.stats["total_retries"] += 1

                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Batch {batch_idx} failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    logger.info(f"Retrying batch {batch_idx} in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Batch {batch_idx} failed after all retries: {e}")
                    self.stats["failed_batches"] += 1
                    raise

        raise last_error or Exception(f"Batch {batch_idx} processing failed")

    def process(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], List[Any]],
        parallel: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """Process items in batches.

        Args:
            items: Items to process
            process_fn: Function that processes a batch of items
            parallel: Whether to process batches in parallel
            progress_callback: Called with (completed, total) batches

        Returns:
            Processed results in the same order as input
        """
        if not items:
            return []

        # Create batches
        batches = chunk_list(items, self.config.batch_size)
        total_batches = len(batches)

        logger.info(f"Processing {len(items)} items in {total_batches} batches")

        # Update statistics
        self.stats["total_batches"] += total_batches
        self.stats["total_items"] += len(items)

        results = []

        if parallel and self.config.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all batches
                future_to_idx = {
                    executor.submit(
                        self._process_batch_with_retry, batch, process_fn, idx
                    ): idx
                    for idx, batch in enumerate(batches)
                }

                # Collect results in order
                batch_results = [None] * total_batches
                completed = 0

                for future in as_completed(future_to_idx, timeout=self.config.timeout):
                    idx = future_to_idx[future]

                    try:
                        batch_results[idx] = future.result()
                        self.stats["successful_batches"] += 1
                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total_batches)

                    except Exception as e:
                        logger.error(f"Batch {idx} failed: {e}")
                        # Return empty results for failed batch
                        batch_results[idx] = [""] * len(batches[idx])
                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total_batches)

                # Flatten results
                for batch_result in batch_results:
                    if batch_result:
                        results.extend(batch_result)

        else:
            # Sequential processing
            for idx, batch in enumerate(batches):
                try:
                    batch_results = self._process_batch_with_retry(
                        batch, process_fn, idx
                    )
                    results.extend(batch_results)
                    self.stats["successful_batches"] += 1

                except Exception as e:
                    logger.error(f"Batch {idx} failed: {e}")
                    # Add empty results for failed batch
                    results.extend([""] * len(batch))

                if progress_callback:
                    progress_callback(idx + 1, total_batches)

        logger.info(f"Completed processing {len(results)} results")
        return results

    def get_stats(self) -> dict:
        """Get processing statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()

        # Calculate success rate
        if stats["total_batches"] > 0:
            stats["success_rate"] = stats["successful_batches"] / stats["total_batches"]
        else:
            stats["success_rate"] = 0.0

        # Calculate average retries per batch
        if stats["total_batches"] > 0:
            stats["avg_retries"] = stats["total_retries"] / stats["total_batches"]
        else:
            stats["avg_retries"] = 0.0

        return stats

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_retries": 0,
            "total_items": 0,
        }


# Convenience function for simple batch processing
def process_batch_with_retry(
    items: List[T],
    process_fn: Callable[[List[T]], List[Any]],
    batch_size: int = 10,
    max_retries: int = 3,
    **kwargs,
) -> List[Any]:
    """Process items in batches with retry logic.

    Args:
        items: Items to process
        process_fn: Function to process a batch
        batch_size: Size of each batch
        max_retries: Maximum retry attempts
        **kwargs: Additional BatchConfig parameters

    Returns:
        Processed results
    """
    config = BatchConfig(batch_size=batch_size, max_retries=max_retries, **kwargs)

    processor = BatchProcessor(config)
    return processor.process(items, process_fn)
