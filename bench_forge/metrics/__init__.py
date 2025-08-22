"""Metrics module for BenchForge."""

from bench_forge.metrics.base import BaseMetric, MetricResult
from bench_forge.metrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
)
from bench_forge.metrics.wrappers import (
    ClassificationMetrics,
    TextMetrics,
    accuracy_score,
    precision_recall_f1,
    confusion_matrix,
    rouge_scores,
    bleu_score,
    text_similarity,
)

__all__ = [
    # Base
    "BaseMetric",
    "MetricResult",
    # Classification
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "ConfusionMatrix",
    # Wrappers
    "ClassificationMetrics",
    "TextMetrics",
    # Helper functions
    "accuracy_score",
    "precision_recall_f1",
    "confusion_matrix",
    "rouge_scores",
    "bleu_score",
    "text_similarity",
]
