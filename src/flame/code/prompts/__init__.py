"""
FLaME Prompt System

This package provides a unified system for managing and accessing prompts
for all FLaME tasks. Prompts are organized by format (zero-shot, few-shot, etc.)
and can be accessed through the registry system.

Usage:
    from flame.code.prompts import get_prompt, PromptFormat

    # Get a prompt function for a specific task and format
    prompt_fn = get_prompt("fpb", PromptFormat.ZERO_SHOT)

    # Use the prompt function with input text
    formatted_prompt = prompt_fn(input_text)

The package includes:
- Base prompt functions for various tasks
- Zero-shot prompt variants
- Few-shot prompt variants
- A registry system for accessing prompts
"""

# Import registry components
from .registry import get_prompt, PromptFormat, register_prompt

# Import all prompt modules to populate the registry
from . import base, zeroshot, fewshot

# Re-export all prompts
__all__ = [
    # Registry components
    "get_prompt",
    "PromptFormat",
    "register_prompt",
    # Base prompts
    "bizbench_prompt",
    "econlogicqa_prompt",
    # Zero-shot prompts
    "headlines_zeroshot_prompt",
    "numclaim_zeroshot_prompt",
    "fomc_zeroshot_prompt",
    "fpb_zeroshot_prompt",
    "banking77_zeroshot_prompt",
    # Few-shot prompts
    "banking77_fewshot_prompt",
    "numclaim_fewshot_prompt",
    "fpb_fewshot_prompt",
    "fomc_fewshot_prompt",
]

# Import specific functions for direct access
from .base import (
    bizbench_prompt,
    econlogicqa_prompt,
)

from .zeroshot import (
    headlines_zeroshot_prompt,
    numclaim_zeroshot_prompt,
    fomc_zeroshot_prompt,
    fpb_zeroshot_prompt,
    banking77_zeroshot_prompt,
    edtsum_zeroshot_prompt,
)

from .fewshot import (
    banking77_fewshot_prompt,
    numclaim_fewshot_prompt,
    fpb_fewshot_prompt,
    fomc_fewshot_prompt,
)

# Define common aliases for backward compatibility with inference_prompts.py
headlines_prompt = headlines_zeroshot_prompt
edtsum_prompt = edtsum_zeroshot_prompt
numclaim_prompt = numclaim_zeroshot_prompt
