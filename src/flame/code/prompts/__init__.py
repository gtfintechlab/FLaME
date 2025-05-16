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
from .registry import get_prompt, PromptFormat, register_prompt, get_prompt_by_name

# Import constants
from .constants import BANKING77_CATEGORIES, FINRED_RELATIONSHIPS

# Import all prompt modules to populate the registry
# These imports are needed to trigger the registry decorators
from . import base  # noqa: F401
from . import zeroshot  # noqa: F401
from . import fewshot  # noqa: F401

# Re-export all prompts
__all__ = [
    # Registry components
    "get_prompt",
    "get_prompt_by_name",
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
    "fiqa_task1_zeroshot_prompt",
    "fiqa_task2_zeroshot_prompt",
    "finer_zeroshot_prompt",
    "finentity_zeroshot_prompt",
    "finbench_zeroshot_prompt",
    "ectsum_zeroshot_prompt",
    "finqa_zeroshot_prompt",
    "convfinqa_zeroshot_prompt",
    "tatqa_zeroshot_prompt",
    "causal_classification_zeroshot_prompt",
    "finred_zeroshot_prompt",
    "causal_detection_zeroshot_prompt",
    "subjectiveqa_zeroshot_prompt",
    "fnxl_zeroshot_prompt",
    "refind_zeroshot_prompt",
    # Few-shot prompts
    "banking77_fewshot_prompt",
    "numclaim_fewshot_prompt",
    "fpb_fewshot_prompt",
    "fomc_fewshot_prompt",
    "headlines_fewshot_prompt",
    "fiqa_task1_fewshot_prompt",
    "fiqa_task2_fewshot_prompt",
    "edtsum_fewshot_prompt",
    "ectsum_fewshot_prompt",
    "finqa_fewshot_prompt",
    "convfinqa_fewshot_prompt",
    "causal_classification_fewshot_prompt",
    "finred_fewshot_prompt",
    "causal_detection_fewshot_prompt",
    "subjectiveqa_fewshot_prompt",
    "fnxl_fewshot_prompt",
    "refind_fewshot_prompt",
    "finentity_fewshot_prompt",
    "finer_fewshot_prompt",
    "finbench_fewshot_prompt",
    # Constants
    "BANKING77_CATEGORIES",
    "FINRED_RELATIONSHIPS",
]

# Import specific functions for direct access
from .base import (
    bizbench_prompt,
    econlogicqa_prompt,
)

from .zeroshot import (
    # Zero-shot prompt functions
    headlines_zeroshot_prompt,
    numclaim_zeroshot_prompt,
    fomc_zeroshot_prompt,
    fpb_zeroshot_prompt,
    banking77_zeroshot_prompt,
    edtsum_zeroshot_prompt,
    fiqa_task1_zeroshot_prompt,
    fiqa_task2_zeroshot_prompt,
    finer_zeroshot_prompt,
    finentity_zeroshot_prompt,
    finbench_zeroshot_prompt,
    ectsum_zeroshot_prompt,
    finqa_zeroshot_prompt,
    convfinqa_zeroshot_prompt,
    tatqa_zeroshot_prompt,
    causal_classification_zeroshot_prompt,
    finred_zeroshot_prompt,
    causal_detection_zeroshot_prompt,
    subjectiveqa_zeroshot_prompt,
    fnxl_zeroshot_prompt,
    refind_zeroshot_prompt,
)

from .fewshot import (
    banking77_fewshot_prompt,
    numclaim_fewshot_prompt,
    fpb_fewshot_prompt,
    fomc_fewshot_prompt,
    headlines_fewshot_prompt,
    fiqa_task1_fewshot_prompt,
    fiqa_task2_fewshot_prompt,
    edtsum_fewshot_prompt,
    ectsum_fewshot_prompt,
    finqa_fewshot_prompt,
    convfinqa_fewshot_prompt,
    causal_classification_fewshot_prompt,
    finred_fewshot_prompt,
    causal_detection_fewshot_prompt,
    subjectiveqa_fewshot_prompt,
    fnxl_fewshot_prompt,
    refind_fewshot_prompt,
    finentity_fewshot_prompt,
    finer_fewshot_prompt,
    finbench_fewshot_prompt,
)

# Define common aliases for backward compatibility with inference_prompts.py
headlines_prompt = headlines_zeroshot_prompt
edtsum_prompt = edtsum_zeroshot_prompt
numclaim_prompt = numclaim_zeroshot_prompt
fiqa_task1_prompt = fiqa_task1_zeroshot_prompt
fiqa_task2_prompt = fiqa_task2_zeroshot_prompt
finer_prompt = finer_zeroshot_prompt
finentity_prompt = finentity_zeroshot_prompt
finbench_prompt = finbench_zeroshot_prompt
ectsum_prompt = ectsum_zeroshot_prompt
finqa_prompt = finqa_zeroshot_prompt
convfinqa_prompt = convfinqa_zeroshot_prompt
tatqa_prompt = tatqa_zeroshot_prompt
causal_classification_prompt = causal_classification_zeroshot_prompt
finred_prompt = finred_zeroshot_prompt
causal_detection_prompt = causal_detection_zeroshot_prompt
subjectiveqa_prompt = subjectiveqa_zeroshot_prompt
fnxl_prompt = fnxl_zeroshot_prompt
refind_prompt = refind_zeroshot_prompt
