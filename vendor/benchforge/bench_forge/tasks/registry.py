"""Task registry with singleton pattern and proper error handling."""

from typing import Any, Dict, List, Optional, Type
import logging
from threading import Lock

from bench_forge.tasks.base import BaseTask
from bench_forge.tasks.config import TaskConfig


logger = logging.getLogger(__name__)


class TaskRegistry:
    """Thread-safe singleton registry for benchmark tasks."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Ensure single instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry (only once)."""
        if self._initialized:
            return

        self._tasks: Dict[str, Type[BaseTask]] = {}
        self._configs: Dict[str, TaskConfig] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._initialized = True
        logger.info("TaskRegistry initialized")

    def register(
        self,
        name: str,
        task_class: Type[BaseTask],
        config: Optional[TaskConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register a task class with optional configuration.

        Args:
            name: Unique task identifier
            task_class: Task class (must inherit from BaseTask)
            config: Optional default configuration
            metadata: Optional task metadata (description, version, etc.)

        Raises:
            TypeError: If task_class doesn't inherit from BaseTask
            ValueError: If name is empty
        """
        if not name:
            raise ValueError("Task name cannot be empty")

        if not issubclass(task_class, BaseTask):
            raise TypeError(f"{task_class} must inherit from BaseTask")

        if name in self._tasks:
            logger.warning(f"Overwriting existing task: {name}")

        self._tasks[name] = task_class

        if config:
            self._configs[name] = config

        if metadata:
            self._metadata[name] = metadata

        logger.info(f"Registered task: {name}")

    def get(self, name: str) -> Type[BaseTask]:
        """Get a registered task class.

        Args:
            name: Task name

        Returns:
            Task class

        Raises:
            KeyError: If task not found
        """
        if name not in self._tasks:
            available = ", ".join(self._tasks.keys())
            raise KeyError(f"Task '{name}' not found. Available tasks: {available}")
        return self._tasks[name]

    def get_config(self, name: str) -> Optional[TaskConfig]:
        """Get default configuration for a task.

        Args:
            name: Task name

        Returns:
            Task configuration or None
        """
        return self._configs.get(name)

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a task.

        Args:
            name: Task name

        Returns:
            Task metadata or empty dict
        """
        return self._metadata.get(name, {})

    def create_task(self, name: str, config: Optional[TaskConfig] = None) -> BaseTask:
        """Create an instance of a registered task.

        Args:
            name: Task name
            config: Configuration to use (overrides default)

        Returns:
            Task instance

        Raises:
            KeyError: If task not found
        """
        task_class = self.get(name)

        # Use provided config or fall back to default
        if config is None:
            config = self.get_config(name)

        # Create instance
        task = task_class(config)

        logger.info(f"Created task instance: {name}")
        return task

    def list_tasks(self) -> List[str]:
        """List all registered task names.

        Returns:
            List of task names
        """
        return list(self._tasks.keys())

    def clear(self):
        """Clear all registered tasks (mainly for testing)."""
        self._tasks.clear()
        self._configs.clear()
        self._metadata.clear()
        logger.info("Registry cleared")

    def __contains__(self, name: str) -> bool:
        """Check if a task is registered."""
        return name in self._tasks

    def __len__(self) -> int:
        """Get number of registered tasks."""
        return len(self._tasks)


# Global registry instance
_registry = TaskRegistry()


def get_registry() -> TaskRegistry:
    """Get the global task registry.

    Returns:
        TaskRegistry instance
    """
    return _registry


def register_task(
    name: str,
    task_class: Optional[Type[BaseTask]] = None,
    config: Optional[TaskConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Register a task with the global registry.

    Can be used as a decorator or function.

    Args:
        name: Task name
        task_class: Task class (if used as function)
        config: Default configuration
        metadata: Task metadata

    Returns:
        Decorator function or None
    """
    registry = get_registry()

    if task_class is not None:
        # Direct registration
        registry.register(name, task_class, config, metadata)
        return task_class

    # Decorator usage
    def decorator(cls: Type[BaseTask]) -> Type[BaseTask]:
        registry.register(name, cls, config, metadata)
        return cls

    return decorator


# Convenience decorator
def task(name: str, **kwargs):
    """Decorator to register a task.

    Args:
        name: Task name
        **kwargs: Additional arguments for register_task

    Returns:
        Decorator function
    """
    return register_task(name, **kwargs)
