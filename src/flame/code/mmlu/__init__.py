"""MMLU (Massive Multitask Language Understanding) evaluation module."""

from flame.code.mmlu.mmlu_constants import (
    ALL_SUBJECTS,
    CHOICES,
    ECONOMICS_SUBJECTS,
    SPLITS,
)
from flame.code.mmlu.mmlu_evaluate import mmlu_evaluate
from flame.code.mmlu.mmlu_inference import mmlu_inference
from flame.code.mmlu.mmlu_loader import MMLULoader

__all__ = [
    "MMLULoader",
    "ECONOMICS_SUBJECTS",
    "ALL_SUBJECTS",
    "SPLITS",
    "CHOICES",
    "mmlu_inference",
    "mmlu_evaluate",
]
