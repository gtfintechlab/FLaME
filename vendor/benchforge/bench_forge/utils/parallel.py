"""Parallel and async execution utilities for BenchForge.

Professional-grade parallel processing with thread pools, process pools,
async support, and intelligent resource management.
"""

import asyncio
import concurrent.futures
import logging
import multiprocessing
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic, Tuple
from queue import Queue, Empty
import threading

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""

    max_workers: Optional[int] = None
    executor_type: str = "thread"  # thread, process, async
    timeout: Optional[float] = None
    chunksize: int = 1
    preserve_order: bool = True
    error_handler: Optional[Callable[[Exception], Any]] = None
    progress_callback: Optional[Callable[[int, int], None]] = None

    def __post_init__(self):
        """Validate and set defaults."""
        if self.max_workers is None:
            if self.executor_type == "thread":
                self.max_workers = min(32, multiprocessing.cpu_count() * 2)
            else:
                self.max_workers = multiprocessing.cpu_count()

        if self.max_workers < 1:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")

        if self.executor_type not in ["thread", "process", "async"]:
            raise ValueError(f"Invalid executor_type: {self.executor_type}")

        if self.chunksize < 1:
            raise ValueError(f"chunksize must be positive, got {self.chunksize}")


@dataclass
class ExecutionResult(Generic[T]):
    """Result of parallel execution."""

    successful: List[T] = field(default_factory=list)
    failed: List[Tuple[int, Exception]] = field(default_factory=list)
    duration: float = 0.0
    num_workers_used: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = len(self.successful) + len(self.failed)
        if total == 0:
            return 1.0
        return len(self.successful) / total

    @property
    def all_successful(self) -> bool:
        """Check if all executions were successful."""
        return len(self.failed) == 0

    def raise_if_failed(self):
        """Raise exception if any execution failed."""
        if self.failed:
            idx, exc = self.failed[0]
            raise RuntimeError(f"Execution failed at index {idx}: {exc}") from exc


class ParallelExecutor:
    """Execute functions in parallel with professional features."""

    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize executor.

        Args:
            config: Parallel execution configuration
        """
        self.config = config or ParallelConfig()
        self._executor = None
        self._stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_duration": 0.0,
            "average_duration": 0.0,
        }
        self._lock = threading.Lock()

        logger.info(
            f"ParallelExecutor initialized: type={self.config.executor_type}, "
            f"max_workers={self.config.max_workers}"
        )

    def __enter__(self):
        """Context manager entry."""
        self._create_executor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup_executor()

    def _create_executor(self):
        """Create the appropriate executor."""
        if self._executor is not None:
            return

        if self.config.executor_type == "thread":
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )
        elif self.config.executor_type == "process":
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.max_workers
            )
        else:
            # Async executor handled separately
            pass

    def _cleanup_executor(self):
        """Cleanup executor resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def map(
        self,
        func: Callable[[T], R],
        items: List[T],
        timeout: Optional[float] = None,
        chunksize: Optional[int] = None,
    ) -> ExecutionResult[R]:
        """Map function over items in parallel.

        Args:
            func: Function to apply
            items: Items to process
            timeout: Timeout for each execution
            chunksize: Chunk size for processing

        Returns:
            ExecutionResult with results and statistics
        """
        if not items:
            return ExecutionResult()

        timeout = timeout or self.config.timeout
        chunksize = chunksize or self.config.chunksize

        start_time = time.time()
        result = ExecutionResult()

        try:
            # Create executor if needed
            self._create_executor()

            if self.config.executor_type == "async":
                # Handle async execution
                return self._map_async(func, items, timeout)

            # Submit all tasks
            futures = []
            for i, item in enumerate(items):
                future = self._executor.submit(func, item)
                futures.append((i, future))

            # Collect results
            completed = 0
            total = len(items)

            for i, future in futures:
                try:
                    value = future.result(timeout=timeout)
                    result.successful.append(value)
                    completed += 1

                    # Progress callback
                    if self.config.progress_callback:
                        self.config.progress_callback(completed, total)

                except Exception as e:
                    logger.error(f"Task {i} failed: {e}")
                    result.failed.append((i, e))

                    # Error handler
                    if self.config.error_handler:
                        try:
                            self.config.error_handler(e)
                        except Exception as handler_error:
                            logger.error(f"Error handler failed: {handler_error}")

            # Preserve order if configured
            if self.config.preserve_order and not result.failed:
                # Results are already in order from successful list
                pass

        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise

        finally:
            result.duration = time.time() - start_time
            result.num_workers_used = self.config.max_workers

            # Update statistics
            self._update_stats(result)

        return result

    def _map_async(
        self, func: Callable[[T], R], items: List[T], timeout: Optional[float]
    ) -> ExecutionResult[R]:
        """Map function over items asynchronously.

        Args:
            func: Function to apply (must be async)
            items: Items to process
            timeout: Timeout for execution

        Returns:
            ExecutionResult
        """
        # Run in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Create async wrapper if func is not async
            if not asyncio.iscoroutinefunction(func):

                async def async_func(item):
                    return func(item)
            else:
                async_func = func

            # Run async execution
            result = loop.run_until_complete(
                self._async_map_impl(async_func, items, timeout)
            )
            return result

        finally:
            loop.close()

    async def _async_map_impl(
        self, func: Callable, items: List[T], timeout: Optional[float]
    ) -> ExecutionResult[R]:
        """Async implementation of map."""
        start_time = time.time()
        result = ExecutionResult()

        # Create tasks
        tasks = []
        for i, item in enumerate(items):
            if timeout:
                task = asyncio.wait_for(func(item), timeout=timeout)
            else:
                task = func(item)
            tasks.append((i, asyncio.create_task(task)))

        # Gather results
        completed = 0
        total = len(items)

        for i, task in tasks:
            try:
                value = await task
                result.successful.append(value)
                completed += 1

                if self.config.progress_callback:
                    self.config.progress_callback(completed, total)

            except Exception as e:
                logger.error(f"Async task {i} failed: {e}")
                result.failed.append((i, e))

                if self.config.error_handler:
                    try:
                        self.config.error_handler(e)
                    except Exception as handler_error:
                        logger.error(f"Error handler failed: {handler_error}")

        result.duration = time.time() - start_time
        result.num_workers_used = self.config.max_workers

        return result

    def starmap(
        self, func: Callable, args_list: List[tuple], timeout: Optional[float] = None
    ) -> ExecutionResult:
        """Apply function with multiple arguments in parallel.

        Args:
            func: Function to apply
            args_list: List of argument tuples
            timeout: Timeout for each execution

        Returns:
            ExecutionResult
        """

        # Create wrapper function
        def wrapper(args):
            return func(*args)

        return self.map(wrapper, args_list, timeout)

    def batch_process(
        self, func: Callable[[List[T]], List[R]], items: List[T], batch_size: int = 100
    ) -> ExecutionResult[R]:
        """Process items in batches.

        Args:
            func: Batch processing function
            items: Items to process
            batch_size: Size of each batch

        Returns:
            ExecutionResult with flattened results
        """
        # Create batches
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        # Process batches
        batch_results = self.map(func, batches)

        # Flatten results
        result = ExecutionResult()
        result.duration = batch_results.duration
        result.num_workers_used = batch_results.num_workers_used

        for batch_result in batch_results.successful:
            result.successful.extend(batch_result)

        result.failed = batch_results.failed

        return result

    def _update_stats(self, result: ExecutionResult):
        """Update execution statistics.

        Args:
            result: Execution result
        """
        with self._lock:
            total_tasks = len(result.successful) + len(result.failed)
            self._stats["total_tasks"] += total_tasks
            self._stats["successful_tasks"] += len(result.successful)
            self._stats["failed_tasks"] += len(result.failed)
            self._stats["total_duration"] += result.duration

            if self._stats["total_tasks"] > 0:
                self._stats["average_duration"] = (
                    self._stats["total_duration"] / self._stats["total_tasks"]
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            return self._stats.copy()


class AsyncExecutor:
    """Async execution utilities with professional features."""

    def __init__(self, max_concurrent: int = 10):
        """Initialize async executor.

        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self.max_concurrent = max_concurrent
        self._semaphore = None
        self._stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_duration": 0.0,
        }

    async def gather(
        self,
        *tasks,
        return_exceptions: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """Gather multiple async tasks with concurrency control.

        Args:
            *tasks: Async tasks to execute
            return_exceptions: Whether to return exceptions as results
            progress_callback: Progress callback function

        Returns:
            List of results
        """
        if not tasks:
            return []

        # Create semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Wrap tasks with semaphore
        wrapped_tasks = [
            self._run_with_semaphore(task, i, len(tasks), progress_callback)
            for i, task in enumerate(tasks)
        ]

        # Gather results
        results = await asyncio.gather(
            *wrapped_tasks, return_exceptions=return_exceptions
        )

        # Update statistics
        self._stats["total_tasks"] += len(tasks)
        self._stats["successful_tasks"] += sum(
            1 for r in results if not isinstance(r, Exception)
        )
        self._stats["failed_tasks"] += sum(
            1 for r in results if isinstance(r, Exception)
        )

        return results

    async def _run_with_semaphore(
        self, task, index: int, total: int, progress_callback: Optional[Callable]
    ):
        """Run task with semaphore control.

        Args:
            task: Async task
            index: Task index
            total: Total number of tasks
            progress_callback: Progress callback

        Returns:
            Task result
        """
        async with self._semaphore:
            start_time = time.time()

            try:
                result = await task

                if progress_callback:
                    progress_callback(index + 1, total)

                duration = time.time() - start_time
                logger.debug(f"Task {index} completed in {duration:.2f}s")

                return result

            except Exception as e:
                logger.error(f"Task {index} failed: {e}")
                raise

    async def map_async(
        self, func: Callable, items: List[Any], preserve_order: bool = True
    ) -> List[Any]:
        """Map async function over items.

        Args:
            func: Async function to apply
            items: Items to process
            preserve_order: Whether to preserve order

        Returns:
            List of results
        """
        # Create tasks
        tasks = [func(item) for item in items]

        # Execute with concurrency control
        results = await self.gather(*tasks)

        return results

    async def batch_async(
        self, func: Callable, items: List[Any], batch_size: int = 100
    ) -> List[Any]:
        """Process items in batches asynchronously.

        Args:
            func: Async batch processing function
            items: Items to process
            batch_size: Batch size

        Returns:
            Flattened results
        """
        # Create batches
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        # Process batches
        batch_results = await self.map_async(func, batches)

        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Statistics dictionary
        """
        return self._stats.copy()


class TaskQueue:
    """Thread-safe task queue for parallel processing."""

    def __init__(self, max_workers: int = 4):
        """Initialize task queue.

        Args:
            max_workers: Number of worker threads
        """
        self.max_workers = max_workers
        self._queue = Queue()
        self._workers = []
        self._stop_event = threading.Event()
        self._results = {}
        self._result_lock = threading.Lock()
        self._task_counter = 0

    def start(self):
        """Start worker threads."""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, args=(i,), daemon=True)
            worker.start()
            self._workers.append(worker)

        logger.info(f"Started {self.max_workers} worker threads")

    def stop(self, timeout: float = 10.0):
        """Stop worker threads.

        Args:
            timeout: Timeout for stopping workers
        """
        # Signal stop
        self._stop_event.set()

        # Add sentinel values to wake up workers
        for _ in range(self.max_workers):
            self._queue.put(None)

        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=timeout)

        self._workers.clear()
        logger.info("Stopped all worker threads")

    def _worker(self, worker_id: int):
        """Worker thread function.

        Args:
            worker_id: Worker identifier
        """
        logger.debug(f"Worker {worker_id} started")

        while not self._stop_event.is_set():
            try:
                # Get task from queue
                task = self._queue.get(timeout=1.0)

                if task is None:
                    # Sentinel value, stop worker
                    break

                task_id, func, args, kwargs = task

                # Execute task
                try:
                    result = func(*args, **kwargs)

                    # Store result
                    with self._result_lock:
                        self._results[task_id] = ("success", result)

                    logger.debug(f"Worker {worker_id} completed task {task_id}")

                except Exception as e:
                    # Store error
                    with self._result_lock:
                        self._results[task_id] = ("error", e)

                    logger.error(f"Worker {worker_id} task {task_id} failed: {e}")

                finally:
                    self._queue.task_done()

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.debug(f"Worker {worker_id} stopped")

    def submit(self, func: Callable, *args, **kwargs) -> int:
        """Submit task to queue.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Task ID
        """
        task_id = self._task_counter
        self._task_counter += 1

        task = (task_id, func, args, kwargs)
        self._queue.put(task)

        return task_id

    def get_result(self, task_id: int, timeout: Optional[float] = None) -> Any:
        """Get task result.

        Args:
            task_id: Task ID
            timeout: Timeout for waiting

        Returns:
            Task result

        Raises:
            TimeoutError: If timeout exceeded
            Exception: If task failed
        """
        start_time = time.time()

        while True:
            with self._result_lock:
                if task_id in self._results:
                    status, value = self._results[task_id]

                    if status == "error":
                        raise value

                    return value

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timeout after {timeout}s")

            time.sleep(0.01)

    def wait_all(self, timeout: Optional[float] = None):
        """Wait for all tasks to complete.

        Args:
            timeout: Timeout for waiting
        """
        self._queue.join()


# Convenience functions
def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    max_workers: Optional[int] = None,
    executor_type: str = "thread",
) -> List[R]:
    """Simple parallel map function.

    Args:
        func: Function to apply
        items: Items to process
        max_workers: Maximum workers
        executor_type: Type of executor

    Returns:
        List of results
    """
    config = ParallelConfig(max_workers=max_workers, executor_type=executor_type)

    with ParallelExecutor(config) as executor:
        result = executor.map(func, items)
        result.raise_if_failed()
        return result.successful


async def async_gather(*tasks, max_concurrent: int = 10) -> List[Any]:
    """Simple async gather with concurrency control.

    Args:
        *tasks: Async tasks
        max_concurrent: Maximum concurrent tasks

    Returns:
        List of results
    """
    executor = AsyncExecutor(max_concurrent=max_concurrent)
    return await executor.gather(*tasks)


# Module exports
__all__ = [
    "ParallelConfig",
    "ExecutionResult",
    "ParallelExecutor",
    "AsyncExecutor",
    "TaskQueue",
    "parallel_map",
    "async_gather",
]
