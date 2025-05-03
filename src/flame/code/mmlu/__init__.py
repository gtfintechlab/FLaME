"""MMLU (Massive Multitask Language Understanding) evaluation module."""

from flame.code.mmlu.mmlu_constants import (
    ECONOMICS_SUBJECTS,
    ALL_SUBJECTS,
    SPLITS,
    CHOICES,
)
from flame.code.mmlu.mmlu_loader import MMLULoader
from flame.code.mmlu.mmlu_inference import mmlu_inference
from flame.code.mmlu.mmlu_evaluate import mmlu_evaluate

__all__ = [
    "MMLULoader",
    "ECONOMICS_SUBJECTS",
    "ALL_SUBJECTS",
    "SPLITS",
    "CHOICES",
    "mmlu_inference",
    "mmlu_evaluate",
]
