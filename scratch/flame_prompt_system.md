# FLaME Prompt System Documentation

This document provides comprehensive documentation for the new prompt system in FLaME. The prompt system was designed to centralize, organize, and standardize how prompts are defined, registered, and accessed throughout the codebase.

## Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Core Components](#core-components)
4. [Using Prompts](#using-prompts)
5. [Creating New Prompts](#creating-new-prompts)
6. [Prompt Registry](#prompt-registry)
7. [Backward Compatibility](#backward-compatibility)
8. [Task and Format Support](#task-and-format-support)
9. [Testing](#testing)
10. [Tips and Best Practices](#tips-and-best-practices)
11. [Migration Guide](#migration-guide)

## Overview

The FLaME prompt system provides a unified interface for creating, registering, and accessing prompts across the entire FLaME framework. It organizes prompts by task (e.g., "headlines", "fpb") and format (e.g., zero-shot, few-shot), and provides a registry-based discovery mechanism.

Key benefits:
- **Centralized management**: All prompts are organized in a single package
- **Format separation**: Different prompt formats are maintained in separate modules
- **Discoverability**: Easy to find and access prompts via registry or imports
- **Extensibility**: Simple to add new prompts or prompt formats
- **Documentation**: Comprehensive docstrings and type annotations
- **Testing**: Improved testability with clear boundaries

## Package Structure

The prompt system is organized into the following structure:

```
src/flame/code/prompts/
├── __init__.py       # Public API and exports
├── base.py           # Default/canonical prompt implementations
├── zeroshot.py       # Zero-shot prompt implementations
├── fewshot.py        # Few-shot prompt implementations
└── registry.py       # Registry system definition
```

Each module serves a specific purpose:
- `__init__.py`: Exports the public API and provides backward compatibility aliases
- `base.py`: Contains default prompt implementations for basic tasks
- `zeroshot.py`: Contains zero-shot prompt implementations for all tasks
- `fewshot.py`: Contains few-shot prompt implementations for all tasks
- `registry.py`: Defines the registry system and PromptFormat enum

## Core Components

### PromptFormat Enum

The `PromptFormat` enum defines the different types of prompts supported by the system:

```python
class PromptFormat(Enum):
    """Enum representing different prompt formats."""
    DEFAULT = auto()
    ZERO_SHOT = auto()
    FEW_SHOT = auto()
    EXTRACTION = auto()
```

### Registry Functions

The registry provides several key functions for registering and accessing prompts:

```python
# Decorator for registering prompts
@register_prompt(task: str, format_type: PromptFormat = PromptFormat.DEFAULT)

# Retrieve a prompt function by task and format
get_prompt(task: str, prompt_format: PromptFormat = PromptFormat.DEFAULT) -> Optional[Callable]

# List all registered tasks and their formats
list_tasks() -> Dict[str, list]

# Get a prompt function by name (backward compatibility)
get_prompt_by_name(prompt_name: str) -> Optional[Callable]
```

## Using Prompts

There are two main ways to use prompts in your code:

### 1. Direct Import (Recommended)

```python
from flame.code.prompts import headlines_zeroshot_prompt, headlines_fewshot_prompt

# Use the prompt functions
zero_shot_prompt = headlines_zeroshot_prompt("Stock market rallies after Fed announcement")
few_shot_prompt = headlines_fewshot_prompt("Stock market rallies after Fed announcement")
```

### 2. Registry Lookup

```python
from flame.code.prompts import get_prompt, PromptFormat

# Get prompt functions by task and format
zero_shot_fn = get_prompt("headlines", PromptFormat.ZERO_SHOT)
few_shot_fn = get_prompt("headlines", PromptFormat.FEW_SHOT)

# Use the retrieved functions
zero_shot_prompt = zero_shot_fn("Stock market rallies after Fed announcement")
few_shot_prompt = few_shot_fn("Stock market rallies after Fed announcement")
```

### Choosing the Right Approach

- **Direct import** is simpler and provides better IDE autocompletion and type checking
- **Registry lookup** is more flexible when the task or format needs to be determined at runtime

## Creating New Prompts

To add a new prompt to the system:

1. Choose the appropriate module (`base.py`, `zeroshot.py`, or `fewshot.py`) based on the prompt format
2. Add your prompt function with proper docstring and type annotations
3. Decorate it with `@register_prompt` to add it to the registry
4. Update `__init__.py` to expose the new function in the public API

Example:

```python
# In zeroshot.py
@register_prompt("my_new_task", PromptFormat.ZERO_SHOT)
def my_new_task_zeroshot_prompt(input_text: str) -> str:
    """Generate a zero-shot prompt for MyNewTask.
    
    Args:
        input_text: The text to include in the prompt
        
    Returns:
        Formatted prompt string
    """
    return f"""Answer the following question about financial data:
    
{input_text}

Your answer:"""
```

Then add to `__init__.py`:

```python
# Add to __all__ list
__all__ = [
    # ... existing entries ...
    "my_new_task_zeroshot_prompt",
]

# Add to the imports section
from .zeroshot import (
    # ... existing imports ...
    my_new_task_zeroshot_prompt,
)
```

## Prompt Registry

The prompt registry maintains a mapping of tasks and formats to their respective prompt functions. This enables:

1. **Runtime discovery**: Find prompts for a task without knowing their import path
2. **Format fallbacks**: Define fallback behavior when a specific format isn't available
3. **Task enumeration**: List all available tasks and formats

### Registry Structure

The registry is implemented as a nested dictionary:

```python
_REGISTRY: Dict[str, Dict[PromptFormat, Callable]] = {}
```

Where:
- Outer keys are task names (e.g., "headlines", "fpb")
- Inner keys are PromptFormat enum values
- Values are callable prompt functions

### Format Fallbacks

When using `get_prompt()` with `PromptFormat.DEFAULT`, the system will try to find an appropriate prompt in this order:

1. `PromptFormat.DEFAULT`
2. `PromptFormat.ZERO_SHOT`
3. `PromptFormat.FEW_SHOT`
4. `PromptFormat.EXTRACTION`

This allows you to request a default prompt for a task without knowing which formats are available.

## Backward Compatibility

The prompt system maintains backward compatibility with the old `prompts_zeroshot.py` and `prompts_fewshot.py` modules through several mechanisms:

1. **Legacy modules**: These files still exist but now import from the new structure
2. **Deprecation warnings**: Users are warned that the old modules will be removed
3. **Aliases**: Common aliases (like `headlines_prompt` = `headlines_zeroshot_prompt`) are provided
4. **Name-based lookup**: `get_prompt_by_name()` allows looking up prompts by their legacy names

The old modules will be removed in a future release, so code should be updated to use the new structure.

## Task and Format Support

The system currently supports the following tasks and formats:

### Supported Formats
- Default
- Zero-shot
- Few-shot
- Extraction (partially implemented)

### Supported Tasks
- banking77
- bizbench
- causal_classification
- causal_detection
- convfinqa
- econlogicqa
- ectsum
- edtsum
- finbench
- finentity
- finer
- finqa
- finred
- fiqa_task1
- fiqa_task2
- fnxl
- fomc
- fpb
- headlines
- numclaim
- refind
- subjectiveqa
- tatqa

Not every task has implementations for all formats. Use `list_tasks()` to see which formats are available for each task.

## Testing

The prompt system includes several tests to ensure proper functionality:

1. **Registry completeness**: Test that all exported prompt functions are registered
2. **Backward compatibility**: Test that old import paths still work
3. **Registry functionality**: Test that the registry correctly returns prompt functions
4. **Documentation**: Test that all prompt functions have proper docstrings

## Tips and Best Practices

### Do's

- Use direct imports when the task and format are known at compile time
- Use the registry when task or format needs to be determined at runtime
- Add comprehensive docstrings to all new prompt functions
- Include proper type annotations
- Update `__init__.py` when adding new prompt functions
- Follow naming conventions: `{task_name}_{format}_prompt`

### Don'ts

- Don't modify existing prompt functions in ways that change their behavior
- Don't add prompts without registering them with `@register_prompt`
- Don't bypass the registration system for new prompts
- Don't create duplicate functions (use aliases in `__init__.py` instead)

## Migration Guide

If you're still using the old prompts modules, follow these steps to migrate:

1. **Update imports**: Change imports from old to new structure

   ```python
   # Old
   from flame.code.prompts_zeroshot import headlines_zeroshot_prompt
   from flame.code.prompts_fewshot import headlines_fewshot_prompt
   
   # New
   from flame.code.prompts import headlines_zeroshot_prompt, headlines_fewshot_prompt
   ```

2. **Update registry usage**: If you're using the global `PROMPT_FUNCTIONS`, switch to the registry

   ```python
   # Old
   from flame.code.prompts_zeroshot import PROMPT_FUNCTIONS
   prompt_func = PROMPT_FUNCTIONS.get("headlines_zeroshot_prompt")
   
   # New
   from flame.code.prompts import get_prompt, PromptFormat
   prompt_func = get_prompt("headlines", PromptFormat.ZERO_SHOT)
   ```

3. **Update prompt creation**: If you're creating new prompts, use the new structure and decorators

   ```python
   # Old (in prompts_zeroshot.py)
   def new_task_zeroshot_prompt(input_text: str) -> str:
       return f"Prompt text: {input_text}"
   
   PROMPT_FUNCTIONS["new_task_zeroshot_prompt"] = new_task_zeroshot_prompt
   
   # New (in prompts/zeroshot.py)
   @register_prompt("new_task", PromptFormat.ZERO_SHOT)
   def new_task_zeroshot_prompt(input_text: str) -> str:
       """Generate a zero-shot prompt for NewTask.
       
       Args:
           input_text: The input text
           
       Returns:
           Formatted prompt string
       """
       return f"Prompt text: {input_text}"
   ```

The old modules (`prompts_zeroshot.py` and `prompts_fewshot.py`) will be removed in a future release, so it's important to update your code to use the new structure as soon as possible.