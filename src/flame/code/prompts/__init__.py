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
from .constants import (
    BANKING77_CATEGORIES,
    FINRED_RELATIONSHIPS,
    banking77_list,
    finred_relationships,
    finred_extraction_labels,
    refind_possible_relationships,
)

# Import all prompt modules to populate the registry
# These imports are needed to trigger the registry decorators
from . import base  # noqa: F401
from . import zeroshot  # noqa: F401
from . import fewshot  # noqa: F401

# Import extraction prompts to register them
from .. import extraction_prompts  # noqa: F401

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
    "banking77_list",
    "finred_relationships",
    "finred_extraction_labels",
    "refind_possible_relationships",
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

"""
BACKWARD COMPATIBILITY ALIASES
==============================

Background:
-----------
The FLaME codebase has transitioned to using a registry-based prompt system
where prompts are accessed via get_prompt(task_name, PromptFormat). However,
for backward compatibility, aliases were created that map shorter names
(e.g., 'numclaim_prompt') to their full names (e.g., 'numclaim_zeroshot_prompt').

Purpose:
--------
These aliases ensure that existing code that imports the shorter prompt names
will continue to work. This was necessary because:
1. External projects might be importing these prompts directly
2. Some scripts or tools might be hardcoded to use these shorter names
3. Breaking these imports would cause downstream failures

TODO:
-----
1. Monitor usage in external projects and determine when these aliases
   can be safely removed
2. Audit the codebase to find any remaining usage of these aliases:
   - Currently, all inference files use the registry system
   - But external projects might still depend on these exports
3. Once confirmed safe, remove these aliases from this file
4. When aliases are removed, also remove test_prompt_aliases.py
5. Update all documentation to reflect the removal

Note: All new code should use get_prompt(task_name, PromptFormat) instead
of importing these aliases directly.
"""

# Define common aliases for backward compatibility
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
