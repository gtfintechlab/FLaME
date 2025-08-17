"""FLAME tasks using BenchForge infrastructure.

This module provides task registration and management for FLAME,
leveraging BenchForge's professional task system.
"""

import logging
from typing import List, Optional

from flame.benchforge import (
    get_registry,
    FLAMEAdapter,
    BENCHFORGE_AVAILABLE,
)

logger = logging.getLogger(__name__)

# Global FLAME adapter
_flame_adapter = None


def get_flame_adapter() -> Optional[FLAMEAdapter]:
    """Get the global FLAME adapter instance.

    Returns:
        FLAMEAdapter instance or None if BenchForge not available
    """
    global _flame_adapter

    if not BENCHFORGE_AVAILABLE:
        logger.warning("BenchForge not available - tasks cannot be registered")
        return None

    if _flame_adapter is None:
        _flame_adapter = FLAMEAdapter()

    return _flame_adapter


def register_all_flame_tasks() -> List[str]:
    """Register all FLAME tasks with BenchForge.

    This function imports all task modules which triggers their @flame_task decorators,
    automatically registering them with BenchForge's task registry.

    Returns:
        List of registered task names
    """
    if not BENCHFORGE_AVAILABLE:
        logger.error(
            "BenchForge is not available. "
            "Please install it: pip install -e ./benchforge"
        )
        return []

    registered_tasks = []

    # Import task modules to trigger registration via @flame_task decorator

    # FOMC task
    try:
        from flame.tasks import fomc  # noqa: F401

        registered_tasks.append("fomc")
        logger.debug("Registered FOMC task")
    except ImportError as e:
        logger.warning(f"Failed to register FOMC: {e}")

    # FPB task
    try:
        from flame.tasks import fpb  # noqa: F401

        registered_tasks.append("fpb")
        logger.debug("Registered FPB task")
    except ImportError as e:
        logger.warning(f"Failed to register FPB: {e}")

    # Banking77 task
    try:
        from flame.tasks import banking77  # noqa: F401

        registered_tasks.append("banking77")
        logger.debug("Registered Banking77 task")
    except ImportError as e:
        logger.debug(f"Banking77 not yet migrated: {e}")

    # Headlines task
    try:
        from flame.tasks import headlines  # noqa: F401

        registered_tasks.append("headlines")
        logger.debug("Registered Headlines task")
    except ImportError as e:
        logger.debug(f"Headlines not yet migrated: {e}")

    # Additional tasks can be added here as they are migrated:
    # - numclaim
    # - finer
    # - finentity
    # - causal_classification
    # - causal_detection
    # - subjectiveqa
    # - ectsum
    # - edtsum
    # - finqa
    # - convfinqa
    # - tatqa
    # - finred
    # - fiqa_task1
    # - fiqa_task2
    # - fnxl
    # - refind
    # - finbench
    # - bizbench

    # Log summary
    logger.info(f"Registered {len(registered_tasks)} FLAME tasks: {registered_tasks}")

    # Verify with BenchForge registry
    registry = get_registry()
    all_tasks = registry.list_tasks()
    logger.debug(f"Total tasks in registry: {len(all_tasks)}")

    return registered_tasks


def list_flame_tasks() -> List[str]:
    """List all available FLAME tasks.

    Returns:
        List of task names
    """
    adapter = get_flame_adapter()
    if adapter:
        return adapter.list_tasks()
    return []


def create_task(name: str, config=None):
    """Create a FLAME task instance.

    Args:
        name: Task name
        config: Optional configuration

    Returns:
        Task instance
    """
    adapter = get_flame_adapter()
    if adapter:
        return adapter.create_task(name, config)

    raise RuntimeError(
        "Cannot create task - BenchForge not available. "
        "Install with: pip install -e ./benchforge"
    )


# Export key functions
__all__ = [
    "register_all_flame_tasks",
    "list_flame_tasks",
    "create_task",
    "get_flame_adapter",
]
