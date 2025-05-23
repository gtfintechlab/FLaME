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


def get_prompt_by_name(prompt_name: str) -> Optional[Callable]:
    """Get a prompt function by its name.

    This function provides backward compatibility with the old prompt_function
    mechanism from prompts_zeroshot.py.

    Args:
        prompt_name: The name of the prompt function

    Returns:
        The prompt function if found, None otherwise
    """
    # Map of old names to new task/format pairs
    name_to_task_format = {
        # Zero-shot prompts
        "headlines_zeroshot_prompt": ("headlines", PromptFormat.ZERO_SHOT),
        "numclaim_zeroshot_prompt": ("numclaim", PromptFormat.ZERO_SHOT),
        "fomc_zeroshot_prompt": ("fomc", PromptFormat.ZERO_SHOT),
        "fpb_zeroshot_prompt": ("fpb", PromptFormat.ZERO_SHOT),
        "banking77_zeroshot_prompt": ("banking77", PromptFormat.ZERO_SHOT),
        "fiqa_task1_zeroshot_prompt": ("fiqa_task1", PromptFormat.ZERO_SHOT),
        "fiqa_task2_zeroshot_prompt": ("fiqa_task2", PromptFormat.ZERO_SHOT),
        "finer_zeroshot_prompt": ("finer", PromptFormat.ZERO_SHOT),
        "finentity_zeroshot_prompt": ("finentity", PromptFormat.ZERO_SHOT),
        "finbench_zeroshot_prompt": ("finbench", PromptFormat.ZERO_SHOT),
        "ectsum_zeroshot_prompt": ("ectsum", PromptFormat.ZERO_SHOT),
        "finqa_zeroshot_prompt": ("finqa", PromptFormat.ZERO_SHOT),
        "convfinqa_zeroshot_prompt": ("convfinqa", PromptFormat.ZERO_SHOT),
        "tatqa_zeroshot_prompt": ("tatqa", PromptFormat.ZERO_SHOT),
        "causal_classification_zeroshot_prompt": (
            "causal_classification",
            PromptFormat.ZERO_SHOT,
        ),
        "finred_zeroshot_prompt": ("finred", PromptFormat.ZERO_SHOT),
        "causal_detection_zeroshot_prompt": (
            "causal_detection",
            PromptFormat.ZERO_SHOT,
        ),
        "subjectiveqa_zeroshot_prompt": ("subjectiveqa", PromptFormat.ZERO_SHOT),
        "fnxl_zeroshot_prompt": ("fnxl", PromptFormat.ZERO_SHOT),
        "refind_zeroshot_prompt": ("refind", PromptFormat.ZERO_SHOT),
        # Aliases without "_zeroshot" suffix
        "headlines_prompt": ("headlines", PromptFormat.ZERO_SHOT),
        "numclaim_prompt": ("numclaim", PromptFormat.ZERO_SHOT),
        "fomc_prompt": ("fomc", PromptFormat.ZERO_SHOT),
        "fpb_prompt": ("fpb", PromptFormat.ZERO_SHOT),
        "fiqa_prompt": ("fiqa_task1", PromptFormat.ZERO_SHOT),
        "edtsum_prompt": ("edtsum", PromptFormat.ZERO_SHOT),
        "ectsum_prompt": ("ectsum", PromptFormat.ZERO_SHOT),
        "finqa_prompt": ("finqa", PromptFormat.ZERO_SHOT),
        "causal_classification_prompt": (
            "causal_classification",
            PromptFormat.ZERO_SHOT,
        ),
        # Few-shot prompts
        "banking77_fewshot_prompt": ("banking77", PromptFormat.FEW_SHOT),
        "numclaim_fewshot_prompt": ("numclaim", PromptFormat.FEW_SHOT),
        "fpb_fewshot_prompt": ("fpb", PromptFormat.FEW_SHOT),
        "fomc_fewshot_prompt": ("fomc", PromptFormat.FEW_SHOT),
    }

    if prompt_name in name_to_task_format:
        task, fmt = name_to_task_format[prompt_name]
        return get_prompt(task, fmt)

    return None
