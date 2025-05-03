"""Utility functions for processing question-answer pairs."""


def process_qa_pair(question: str, answer: str) -> str:
    """Process a single question-answer pair.

    Args:
        question: The question text
        answer: The answer text

    Returns:
        Formatted QA pair string
    """
    return f"Q: {question}\nA: {answer}"
