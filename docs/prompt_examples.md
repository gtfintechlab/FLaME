# FLaME Prompt System Examples

This document provides examples of how to use the new FLaME prompt system for common scenarios.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Using the Registry](#using-the-registry)
3. [Creating New Prompts](#creating-new-prompts)
4. [Switching Between Formats](#switching-between-formats)
5. [Extending the System](#extending-the-system)

## Basic Usage

### Direct Prompt Usage

The simplest way to use prompts is to import them directly from the `flame.code.prompts` package:

```python
from flame.code.prompts import headlines_zeroshot_prompt

# Use the prompt function with input text
prompt = headlines_zeroshot_prompt("Stock market soars on Federal Reserve announcement")
print(prompt)
```

This will produce a fully formatted prompt ready to be sent to an LLM.

### Using Multiple Prompt Formats

When a task supports multiple prompt formats, you can import and use them side by side:

```python
from flame.code.prompts import headlines_zeroshot_prompt, headlines_fewshot_prompt

# Example input text
headline = "Stock market soars on Federal Reserve announcement"

# Generate prompts in different formats
zero_shot = headlines_zeroshot_prompt(headline)
few_shot = headlines_fewshot_prompt(headline)

print("Zero-shot prompt:")
print(zero_shot)
print("\nFew-shot prompt:")
print(few_shot)
```

## Using the Registry

### Getting Prompts by Task and Format

The registry allows you to look up prompt functions dynamically:

```python
from flame.code.prompts import get_prompt, PromptFormat

# Get prompt functions for different formats of the same task
zero_shot_fn = get_prompt("headlines", PromptFormat.ZERO_SHOT)
few_shot_fn = get_prompt("headlines", PromptFormat.FEW_SHOT)

# Example input text
headline = "Stock market soars on Federal Reserve announcement"

# Use the retrieved functions
zero_shot_prompt = zero_shot_fn(headline)
few_shot_prompt = few_shot_fn(headline)

print("Zero-shot prompt:")
print(zero_shot_prompt)
print("\nFew-shot prompt:")
print(few_shot_prompt)
```

### Dynamically Selecting Format Based on User Input

When the format needs to be selected at runtime:

```python
from flame.code.prompts import get_prompt, PromptFormat

def generate_prompt(task: str, input_text: str, use_few_shot: bool = False):
    """Generate a prompt for the given task and input text."""
    # Determine the format based on the use_few_shot flag
    format_type = PromptFormat.FEW_SHOT if use_few_shot else PromptFormat.ZERO_SHOT
    
    # Get the prompt function from the registry
    prompt_fn = get_prompt(task, format_type)
    
    if prompt_fn is None:
        raise ValueError(f"No prompt found for task '{task}' with format '{format_type.name}'")
    
    # Generate and return the prompt
    return prompt_fn(input_text)

# Example usage
tasks = ["headlines", "fpb", "banking77"]
for task in tasks:
    try:
        # Try both zero-shot and few-shot
        print(f"Task: {task} (Zero-shot)")
        print(generate_prompt(task, "Example input text"))
        print(f"\nTask: {task} (Few-shot)")
        print(generate_prompt(task, "Example input text", use_few_shot=True))
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 50)
```

### Listing Available Tasks and Formats

To discover what tasks and formats are available:

```python
from flame.code.prompts import list_tasks

# Get a dictionary of all tasks and their available formats
tasks_and_formats = list_tasks()

# Print the available tasks and formats
print("Available Tasks and Formats:")
for task, formats in tasks_and_formats.items():
    format_str = ", ".join(formats)
    print(f"- {task}: {format_str}")
```

## Creating New Prompts

### Adding a New Zero-Shot Prompt

To add a new zero-shot prompt:

```python
from flame.code.prompts import register_prompt, PromptFormat

@register_prompt("market_analysis", PromptFormat.ZERO_SHOT)
def market_analysis_zeroshot_prompt(text: str) -> str:
    """Generate a zero-shot prompt for market analysis.
    
    Args:
        text: The market data or description to analyze
        
    Returns:
        Formatted prompt string
    """
    return f"""Analyze the following market data and provide insights:

{text}

Your analysis should include:
1. Key trends
2. Potential market movements
3. Risk factors
4. Investment opportunities

Analysis:"""
```

### Adding a New Few-Shot Prompt

To add a new few-shot prompt:

```python
from flame.code.prompts import register_prompt, PromptFormat

@register_prompt("market_analysis", PromptFormat.FEW_SHOT)
def market_analysis_fewshot_prompt(text: str) -> str:
    """Generate a few-shot prompt for market analysis.
    
    Args:
        text: The market data or description to analyze
        
    Returns:
        Formatted prompt string
    """
    # Define a few examples
    examples = [
        {
            "input": "S&P 500 dropped 2% amid concerns about inflation data.",
            "output": "Analysis:\n1. The market is responding to inflation concerns\n2. This could signal a shift in Fed policy\n3. Risk: continued inflation could lead to further drops\n4. Opportunity: defensive sectors may outperform"
        },
        {
            "input": "Tech stocks rally after positive earnings reports from major companies.",
            "output": "Analysis:\n1. Tech sector showing resilience despite macro concerns\n2. Earnings growth remains strong in selective companies\n3. Risk: valuations remain elevated compared to historical norms\n4. Opportunity: quality tech companies with strong cash flows"
        }
    ]
    
    # Format the examples
    examples_text = ""
    for i, example in enumerate(examples, 1):
        examples_text += f"Example {i}:\nData: {example['input']}\n{example['output']}\n\n"
    
    # Build the complete prompt
    return f"""Analyze the following market data and provide insights.
Here are some examples of market analysis:

{examples_text}
Now analyze this new data:

Data: {text}

Analysis:"""
```

## Switching Between Formats

### Implementing Format Selection in Inference Modules

For inference modules that need to support multiple prompt formats:

```python
from flame.code.prompts import get_prompt, PromptFormat

def run_inference(task: str, inputs: list, prompt_format: str = "zero_shot"):
    """Run inference for a given task and inputs."""
    # Map string format to enum
    format_map = {
        "zero_shot": PromptFormat.ZERO_SHOT,
        "few_shot": PromptFormat.FEW_SHOT,
        "default": PromptFormat.DEFAULT
    }
    
    # Get the format enum (default to zero-shot if not found)
    format_enum = format_map.get(prompt_format.lower(), PromptFormat.ZERO_SHOT)
    
    # Get the prompt function from the registry
    prompt_fn = get_prompt(task, format_enum)
    
    if prompt_fn is None:
        raise ValueError(f"No prompt found for task '{task}' with format '{prompt_format}'")
    
    # Generate prompts for all inputs
    prompts = [prompt_fn(input_text) for input_text in inputs]
    
    # Now use these prompts with your LLM API
    # ...
    
    return prompts
```

### Command-Line Format Selection

For main scripts that take format as a command-line argument:

```python
import argparse
from flame.code.prompts import get_prompt, PromptFormat

def main():
    parser = argparse.ArgumentParser(description="Run inference with FLaME prompts")
    parser.add_argument("--task", required=True, help="Task name (e.g., headlines, fpb)")
    parser.add_argument("--input", required=True, help="Input text")
    parser.add_argument("--format", choices=["zero_shot", "few_shot", "default"],
                        default="zero_shot", help="Prompt format")
    args = parser.parse_args()
    
    # Map string format to enum
    format_map = {
        "zero_shot": PromptFormat.ZERO_SHOT,
        "few_shot": PromptFormat.FEW_SHOT,
        "default": PromptFormat.DEFAULT
    }
    format_enum = format_map[args.format]
    
    # Get the prompt function
    prompt_fn = get_prompt(args.task, format_enum)
    
    if prompt_fn is None:
        print(f"Error: No prompt found for task '{args.task}' with format '{args.format}'")
        return 1
    
    # Generate and print the prompt
    prompt = prompt_fn(args.input)
    print(prompt)
    
    return 0

if __name__ == "__main__":
    exit(main())
```

## Extending the System

### Adding a New Prompt Format

To add a new prompt format to the system:

1. First, update the `PromptFormat` enum in `registry.py`:

```python
class PromptFormat(Enum):
    """Enum representing different prompt formats."""
    DEFAULT = auto()
    ZERO_SHOT = auto()
    FEW_SHOT = auto()
    EXTRACTION = auto()
    CHAIN_OF_THOUGHT = auto()  # New format added
```

2. Create a new module for the format (e.g., `chain_of_thought.py`):

```python
"""
Chain of Thought Prompt Functions

This module contains prompt functions that guide the model through a step-by-step
reasoning process before arriving at the final answer.
"""

from .registry import register_prompt, PromptFormat

@register_prompt("headlines", PromptFormat.CHAIN_OF_THOUGHT)
def headlines_cot_prompt(headline: str) -> str:
    """Generate a chain-of-thought prompt for headline sentiment classification.
    
    Args:
        headline: The financial headline to classify
        
    Returns:
        Formatted prompt string
    """
    return f"""Classify the sentiment of the following financial headline.
First, break down the meaning of the headline. Then analyze the financial implications.
Finally, determine if the sentiment is positive, negative, or neutral.

Headline: {headline}

Reasoning:
1. What does this headline mean in plain language?
2. What financial implications does this news have?
3. Who might benefit or be harmed by this news?
4. Given the above analysis, what is the overall sentiment?

Now provide your step-by-step reasoning and final classification:"""
```

3. Update `__init__.py` to include the new module and its functions:

```python
# Import the new module
from . import chain_of_thought

# Add to __all__
__all__ = [
    # ... existing entries ...
    "headlines_cot_prompt",
]

# Import specific functions
from .chain_of_thought import (
    headlines_cot_prompt,
)
```

### Creating Task-Specific Prompt Variants

For specialized tasks that need variant prompts:

```python
from flame.code.prompts import register_prompt, PromptFormat

# Register with a more specific task name
@register_prompt("headlines_financial", PromptFormat.ZERO_SHOT)
def headlines_financial_zeroshot_prompt(headline: str) -> str:
    """Generate a zero-shot prompt for financial headlines classification.
    
    This is a specialized version focusing specifically on financial implications.
    
    Args:
        headline: The financial headline to classify
        
    Returns:
        Formatted prompt string
    """
    return f"""Classify the sentiment of the following financial headline,
focusing specifically on its implications for financial markets and investments.

Headline: {headline}

The sentiment should be classified as:
- Positive: Likely to have a positive impact on markets or specific investments
- Negative: Likely to have a negative impact on markets or specific investments
- Neutral: Likely to have minimal or mixed impacts on markets

Sentiment:"""
```

These variants can be accessed through the registry using their specific task names.