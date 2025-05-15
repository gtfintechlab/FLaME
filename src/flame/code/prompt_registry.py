"""Unified registry for all prompt functions in the FLaME framework.

This module provides a centralized registry for all prompt functions used across
different tasks in the FLaME framework. It helps to:

1. Maintain a single source of truth for which prompt function to use
2. Support different prompt formats (zero-shot, few-shot, etc.)
3. Eliminate duplication and ensure consistency

Usage:
    # Get a prompt function
    from flame.code.prompt_registry import get_prompt, PromptFormat

    # Get a zero-shot prompt function for a task
    prompt_fn = get_prompt("fpb", PromptFormat.ZERO_SHOT)

    # Use the prompt function
    formatted_prompt = prompt_fn(input_text)
"""

from enum import Enum
from typing import Callable, Dict, Optional

# Import all prompt functions from their current locations
# This maintains backward compatibility while centralizing registry
import inspect
from flame.code import (
    inference_prompts,
    prompts,
    prompts_zeroshot,
    prompts_fewshot,
    prompts_fromferrari,
    extraction_prompts,
)

# Import all functions to the current namespace
# We need to do this explicitly to handle module-level registration correctly
for module in [
    inference_prompts,
    prompts,
    prompts_zeroshot,
    prompts_fewshot,
    prompts_fromferrari,
    extraction_prompts,
]:
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.endswith("_prompt") and not name.startswith("_"):
            globals()[name] = func


class PromptFormat(Enum):
    """Enum for different prompt formats."""

    ZERO_SHOT = "zeroshot"
    FEW_SHOT = "fewshot"
    EXTRACTION = "extraction"
    DEFAULT = "default"  # For backward compatibility


# Registry that maps task names to prompt functions for different formats
_REGISTRY: Dict[str, Dict[PromptFormat, Callable]] = {}


def register(task: str, prompt_format: PromptFormat = PromptFormat.DEFAULT) -> Callable:
    """Register a prompt function for a task and format.

    Args:
        task: The task name (e.g., "fpb", "numclaim")
        prompt_format: The prompt format (zero-shot, few-shot, etc.)

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        if task not in _REGISTRY:
            _REGISTRY[task] = {}
        _REGISTRY[task][prompt_format] = func
        return func

    return decorator


def get_prompt(
    task: str, prompt_format: PromptFormat = PromptFormat.DEFAULT
) -> Optional[Callable]:
    """Get a prompt function for a task and format.

    Args:
        task: The task name (e.g., "fpb", "numclaim")
        prompt_format: The prompt format (zero-shot, few-shot, etc.)

    Returns:
        The prompt function or None if not found
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
        ]:
            if format_priority in _REGISTRY[task]:
                return _REGISTRY[task][format_priority]

    # If not found, return None
    return None


# Register all prompt functions
# This is a temporary measure until we migrate all prompt functions
# to use the @register decorator


# Helper to build the registry from existing functions
def _populate_registry():
    """Populate the registry with existing prompt functions."""
    # Directly process all modules to ensure we capture every function
    for module in [
        inference_prompts,
        prompts,
        prompts_zeroshot,
        prompts_fewshot,
        prompts_fromferrari,
        extraction_prompts,
    ]:
        for name, func in inspect.getmembers(module, inspect.isfunction):
            # Skip functions that aren't prompt functions
            if not name.endswith("_prompt") or name.startswith("_"):
                continue

            # Skip the duplicate finqa_prompt from inference_prompts.py
            if name == "finqa_prompt" and module == inference_prompts:
                continue

            # Determine task and format type
            if name.endswith("_zeroshot_prompt"):
                task_name = name[: -len("_zeroshot_prompt")]
                format_type = PromptFormat.ZERO_SHOT
            elif name.endswith("_fewshot_prompt"):
                task_name = name[: -len("_fewshot_prompt")]
                format_type = PromptFormat.FEW_SHOT
            elif name.endswith("_extraction_prompt"):
                task_name = name[: -len("_extraction_prompt")]
                format_type = PromptFormat.EXTRACTION
            else:
                # Default prompts like banking77_prompt
                task_name = name[: -len("_prompt")]
                format_type = PromptFormat.DEFAULT

                # Special case for zero-shot prompts in prompts_zeroshot.py that don't follow the naming convention
                if module == prompts_zeroshot and task_name in ["numclaim", "fomc"]:
                    format_type = PromptFormat.ZERO_SHOT

            # Add to registry
            if task_name not in _REGISTRY:
                _REGISTRY[task_name] = {}

            _REGISTRY[task_name][format_type] = func

            # For backward compatibility, also register zero-shot as DEFAULT if no DEFAULT exists
            if (
                format_type == PromptFormat.ZERO_SHOT
                and PromptFormat.DEFAULT not in _REGISTRY[task_name]
            ):
                _REGISTRY[task_name][PromptFormat.DEFAULT] = func

    # Special case: Special handling for numclaim_prompt in prompts_zeroshot.py
    # This function is not named numclaim_zeroshot_prompt, but it should still be registered as zero-shot
    for func in [
        prompts_zeroshot.numclaim_prompt,
        prompts_zeroshot.fomc_zeroshot_prompt,
    ]:
        task_name = func.__name__.replace("_zeroshot_prompt", "").replace("_prompt", "")
        _REGISTRY[task_name][PromptFormat.ZERO_SHOT] = func


# Populate the registry with existing functions
_populate_registry()


# Function to list all available tasks
def list_tasks() -> Dict[str, Dict[str, str]]:
    """List all available tasks and their prompt formats.

    Returns:
        Dictionary mapping task names to available formats and function names
    """
    result = {}
    for task, formats in _REGISTRY.items():
        result[task] = {}
        for format_type, func in formats.items():
            result[task][format_type.value] = func.__name__
    return result
