"""
Unified Registry for FLaME Prompts

This module provides a centralized registry for all prompt functions across
the FLaME system. It allows prompt functions to be registered with their
task name and format, and retrieved using a common interface.
"""

from enum import Enum, auto
from typing import Dict, Callable, Optional


class PromptFormat(Enum):
    """Enum representing different prompt formats."""

    DEFAULT = auto()
    ZERO_SHOT = auto()
    FEW_SHOT = auto()
    EXTRACTION = auto()


# Global registry of all prompt functions
_REGISTRY: Dict[str, Dict[PromptFormat, Callable]] = {}


def register_prompt(task: str, format_type: PromptFormat = PromptFormat.DEFAULT):
    """Decorator to register a prompt function with the registry.

    Args:
        task: The task name (e.g. "fpb", "headlines")
        format_type: The prompt format (DEFAULT, ZERO_SHOT, etc.)

    Returns:
        Decorator function that registers the decorated function
    """

    def decorator(func: Callable) -> Callable:
        # Initialize task entry if it doesn't exist
        if task not in _REGISTRY:
            _REGISTRY[task] = {}

        # Register function
        _REGISTRY[task][format_type] = func

        # Return original function unchanged
        return func

    return decorator


def get_prompt(
    task: str, prompt_format: PromptFormat = PromptFormat.DEFAULT
) -> Optional[Callable]:
    """Get a prompt function for a task and format.

    Args:
        task: The task name (e.g., "fpb", "headlines")
        prompt_format: The desired prompt format

    Returns:
        The prompt function if found, None otherwise
    """
    # First, try to get the exact format requested
    if task in _REGISTRY and prompt_format in _REGISTRY[task]:
        return _REGISTRY[task][prompt_format]

    # If not found and format is DEFAULT, try to find any format
    if prompt_format == PromptFormat.DEFAULT and task in _REGISTRY:
        # Priority: DEFAULT, ZERO_SHOT, FEW_SHOT, EXTRACTION
        for format_priority in [
            PromptFormat.DEFAULT,
            PromptFormat.ZERO_SHOT,
            PromptFormat.FEW_SHOT,
            PromptFormat.EXTRACTION,
        ]:
            if format_priority in _REGISTRY[task]:
                return _REGISTRY[task][format_priority]

    # Not found
    return None


def list_tasks() -> Dict[str, list]:
    """List all registered tasks and their available formats.

    Returns:
        Dictionary of task names mapped to lists of available format names
    """
    result = {}
    for task, formats in _REGISTRY.items():
        result[task] = []
        for fmt in formats:
            # Convert enum name to lowercase for consistency
            fmt_name = fmt.name.lower()
            result[task].append(fmt_name)
    return result


# Add automatic import and registration for all prompt modules
# This functionality will be activated once we have migrated all prompt functions

# Import all prompt modules to populate the registry
# Will be uncommented once we have completed the migration
# from . import base, zeroshot, fewshot
