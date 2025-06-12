"""
Base Prompt Functions

This module contains the default prompt functions for various FLaME tasks.
These are often the canonical implementations that are most widely used.
"""

from .registry import PromptFormat, register_prompt


@register_prompt("bizbench", PromptFormat.DEFAULT)
def bizbench_prompt(query: str) -> str:
    """Generate a prompt for the BizBench task.

    This prompt asks the model to answer a challenging business question.

    Args:
        query: The business question to answer

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are answering a challenging question about business.
Please provide a detailed, accurate, and helpful answer to the following business question:

{query}

Answer:"""
    return prompt


@register_prompt("econlogicqa", PromptFormat.DEFAULT)
def econlogicqa_prompt(query: str) -> str:
    """Generate a prompt for the EconLogicQA task.

    This prompt asks the model to answer an economics question with logical reasoning.

    Args:
        query: The economics question to answer

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are answering a challenging question about economics.
Please provide a detailed, step-by-step logical answer to the following economics question:

{query}

Answer:"""
    return prompt


# We'll migrate more base prompts as we continue implementation
# This is just a starting point with two clearly defined default prompts
