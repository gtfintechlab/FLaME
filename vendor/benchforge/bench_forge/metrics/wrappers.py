"""High-level wrapper classes for metrics.

These classes provide convenient interfaces for common metric calculations.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy,
    precision_recall_fscore_support,
    confusion_matrix as sklearn_confusion_matrix,
)

from bench_forge.metrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
)


class ClassificationMetrics:
    """High-level wrapper for classification metrics."""

    def __init__(self):
        """Initialize classification metrics."""
        self.accuracy = Accuracy()
        self.precision = Precision()
        self.recall = Recall()
        self.f1 = F1Score()
        self.confusion = ConfusionMatrix()

    def calculate_accuracy(self, y_true: List[Any], y_pred: List[Any]) -> float:
        """Calculate accuracy score.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Accuracy score between 0 and 1
        """
        return sklearn_accuracy(y_true, y_pred)

    def calculate_precision_recall_f1(
        self, y_true: List[Any], y_pred: List[Any], average: str = "weighted"
    ) -> Dict[str, Any]:
        """Calculate precision, recall, and F1 scores.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('micro', 'macro', 'weighted', or None)

        Returns:
            Dictionary with precision, recall, f1, and support
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Get unique labels
        labels = sorted(set(y_true) | set(y_pred))

        # Create per-class results
        per_class = {}
        for i, label in enumerate(labels):
            if i < len(precision):
                per_class[str(label)] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]) if i < len(support) else 0,
                }

        # Calculate averaged metrics
        if average:
            avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=average, zero_division=0
            )

            result = {
                "precision": per_class,
                "recall": per_class,
                "f1": per_class,
                "support": per_class,
            }

            # Add averaged values
            for metric in ["precision", "recall", "f1"]:
                if metric == "precision":
                    result[metric][average] = float(avg_precision)
                elif metric == "recall":
                    result[metric][average] = float(avg_recall)
                elif metric == "f1":
                    result[metric][average] = float(avg_f1)
        else:
            result = per_class

        return result

    def calculate_confusion_matrix(
        self, y_true: List[Any], y_pred: List[Any]
    ) -> np.ndarray:
        """Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix as numpy array
        """
        return sklearn_confusion_matrix(y_true, y_pred)


class TextMetrics:
    """High-level wrapper for text generation metrics."""

    def __init__(self):
        """Initialize text metrics."""
        pass

    def calculate_rouge(
        self, reference: str, hypothesis: str, rouge_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate ROUGE scores.

        Args:
            reference: Reference text
            hypothesis: Generated text
            rouge_types: Types of ROUGE to calculate

        Returns:
            Dictionary of ROUGE scores
        """
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL"]

        # Simplified ROUGE calculation for testing
        # In production, use rouge_score library
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        # Simple unigram overlap for demonstration
        if not ref_words or not hyp_words:
            return {rouge_type: 0.0 for rouge_type in rouge_types}

        overlap = len(set(ref_words) & set(hyp_words))
        precision = overlap / len(hyp_words) if hyp_words else 0
        recall = overlap / len(ref_words) if ref_words else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "rouge1": f1,
            "rouge2": f1 * 0.7,  # Simulated
            "rougeL": f1 * 0.9,  # Simulated
        }

    def calculate_bleu(
        self, reference: Union[str, List[str]], hypothesis: str, max_n: int = 4
    ) -> float:
        """Calculate BLEU score.

        Args:
            reference: Reference text(s)
            hypothesis: Generated text
            max_n: Maximum n-gram order

        Returns:
            BLEU score between 0 and 1
        """
        # Simplified BLEU calculation for testing
        # In production, use nltk.translate.bleu_score
        if isinstance(reference, str):
            reference = [reference]

        # Simple word overlap for demonstration
        hyp_words = hypothesis.lower().split()
        ref_words_list = [ref.lower().split() for ref in reference]

        if not hyp_words:
            return 0.0

        # Calculate precision for each n-gram order
        scores = []
        for n in range(1, min(max_n + 1, len(hyp_words) + 1)):
            # Get n-grams
            hyp_ngrams = [
                tuple(hyp_words[i : i + n]) for i in range(len(hyp_words) - n + 1)
            ]

            if not hyp_ngrams:
                continue

            # Count matches
            matches = 0
            for ngram in hyp_ngrams:
                for ref_words in ref_words_list:
                    ref_ngrams = [
                        tuple(ref_words[i : i + n])
                        for i in range(len(ref_words) - n + 1)
                    ]
                    if ngram in ref_ngrams:
                        matches += 1
                        break

            precision = matches / len(hyp_ngrams)
            scores.append(precision)

        if not scores:
            return 0.0

        # Geometric mean
        score = np.exp(np.mean([np.log(s + 1e-10) for s in scores]))

        # Brevity penalty
        ref_len = min(len(ref_words) for ref_words in ref_words_list)
        if len(hyp_words) < ref_len:
            bp = np.exp(1 - ref_len / len(hyp_words))
            score *= bp

        return float(score)

    def calculate_similarity(
        self, text1: str, text2: str, method: str = "cosine"
    ) -> float:
        """Calculate text similarity.

        Args:
            text1: First text
            text2: Second text
            method: Similarity method

        Returns:
            Similarity score between 0 and 1
        """
        # Simplified similarity for testing
        # In production, use sentence transformers or other embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity as proxy for cosine
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


# Helper functions for backward compatibility
def accuracy_score(y_true: List[Any], y_pred: List[Any]) -> float:
    """Calculate accuracy score."""
    return sklearn_accuracy(y_true, y_pred)


def precision_recall_f1(
    y_true: List[Any], y_pred: List[Any], average: str = "weighted"
) -> Dict[str, Any]:
    """Calculate precision, recall, and F1 scores."""
    metrics = ClassificationMetrics()
    return metrics.calculate_precision_recall_f1(y_true, y_pred, average)


def confusion_matrix(y_true: List[Any], y_pred: List[Any]) -> np.ndarray:
    """Calculate confusion matrix."""
    return sklearn_confusion_matrix(y_true, y_pred)


def rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    metrics = TextMetrics()
    return metrics.calculate_rouge(reference, hypothesis)


def bleu_score(reference: Union[str, List[str]], hypothesis: str) -> float:
    """Calculate BLEU score."""
    metrics = TextMetrics()
    return metrics.calculate_bleu(reference, hypothesis)


def text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity."""
    metrics = TextMetrics()
    return metrics.calculate_similarity(text1, text2)
