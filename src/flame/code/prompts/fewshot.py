"""
Few-Shot Prompt Functions

This module contains few-shot prompt functions for various FLaME tasks.
Few-shot prompts include examples to help guide the model's behavior.
"""

from .registry import PromptFormat, register_prompt

# The following are stub functions that will be implemented in the future
# They are included here for completeness and to maintain registry consistency


@register_prompt("banking77", PromptFormat.FEW_SHOT)
def banking77_fewshot_prompt(sentence: str) -> str:
    """Generate a few-shot prompt for the Banking77 task.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to classify

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("numclaim", PromptFormat.FEW_SHOT)
def numclaim_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot NumClaim prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to classify

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("fpb", PromptFormat.FEW_SHOT)
def fpb_fewshot_prompt(sentence: str, prompt_format: str = None) -> str:
    """Stub for few-shot FPB prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The financial sentence to classify
        prompt_format: Optional format specifier

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("fomc", PromptFormat.FEW_SHOT)
def fomc_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot FOMC prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The FOMC sentence to classify

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("headlines", PromptFormat.FEW_SHOT)
def headlines_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot Headlines prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The headline to classify

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("fiqa_task1", PromptFormat.FEW_SHOT)
def fiqa_task1_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot FiQA Task 1 prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The financial sentence to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("fiqa_task2", PromptFormat.FEW_SHOT)
def fiqa_task2_fewshot_prompt(question: str) -> str:
    """Stub for few-shot FiQA Task 2 prompt.

    This is a placeholder for future implementation.

    Args:
        question: The financial question to answer

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("edtsum", PromptFormat.FEW_SHOT)
def edtsum_fewshot_prompt(document: str) -> str:
    """Stub for few-shot EDTSum prompt.

    This is a placeholder for future implementation.

    Args:
        document: The document to summarize

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("ectsum", PromptFormat.FEW_SHOT)
def ectsum_fewshot_prompt(document: str) -> str:
    """Stub for few-shot ECTSum prompt.

    This is a placeholder for future implementation.

    Args:
        document: The document to summarize

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("finqa", PromptFormat.FEW_SHOT)
def finqa_fewshot_prompt(document: str) -> str:
    """Stub for few-shot FinQA prompt.

    This is a placeholder for future implementation.

    Args:
        document: The financial document to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("convfinqa", PromptFormat.FEW_SHOT)
def convfinqa_fewshot_prompt(document: str) -> str:
    """Stub for few-shot ConvFinQA prompt.

    This is a placeholder for future implementation.

    Args:
        document: The conversational financial document to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("causal_classification", PromptFormat.FEW_SHOT)
def causal_classification_fewshot_prompt(text: str) -> str:
    """Stub for few-shot Causal Classification prompt.

    This is a placeholder for future implementation.

    Args:
        text: The text to classify

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("finred", PromptFormat.FEW_SHOT)
def finred_fewshot_prompt(sentence: str, entity1: str, entity2: str) -> str:
    """Stub for few-shot FinRED prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to analyze
        entity1: The first entity
        entity2: The second entity

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("causal_detection", PromptFormat.FEW_SHOT)
def causal_detection_fewshot_prompt(tokens: list) -> str:
    """Stub for few-shot Causal Detection prompt.

    This is a placeholder for future implementation.

    Args:
        tokens: The list of tokens to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("subjectiveqa", PromptFormat.FEW_SHOT)
def subjectiveqa_fewshot_prompt(feature, definition, question, answer) -> str:
    """Stub for few-shot SubjectiveQA prompt.

    This is a placeholder for future implementation.

    Args:
        feature: The feature to analyze
        definition: The definition of the feature
        question: The question to answer
        answer: The answer to evaluate

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("fnxl", PromptFormat.FEW_SHOT)
def fnxl_fewshot_prompt(sentence, company, doc_type) -> str:
    """Stub for few-shot FNXL prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to analyze
        company: The company name
        doc_type: The document type

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("refind", PromptFormat.FEW_SHOT)
def refind_fewshot_prompt(entities) -> str:
    """Stub for few-shot ReFinD prompt.

    This is a placeholder for future implementation.

    Args:
        entities: The entities to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("finentity", PromptFormat.FEW_SHOT)
def finentity_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot FinEntity prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("finer", PromptFormat.FEW_SHOT)
def finer_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot FinER prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("finbench", PromptFormat.FEW_SHOT)
def finbench_fewshot_prompt(profile: str) -> str:
    """Stub for few-shot FinBench prompt.

    This is a placeholder for future implementation.

    Args:
        profile: The profile to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None
