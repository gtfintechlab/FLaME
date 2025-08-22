"""Classification metrics implementation."""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from bench_forge.metrics.base import BaseMetric


logger = logging.getLogger(__name__)


# Try to import sklearn for advanced metrics
try:
    from sklearn import metrics as sklearn_metrics

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning(
        "scikit-learn not available, some metrics will use basic implementations"
    )


class Accuracy(BaseMetric):
    """Accuracy metric for classification."""

    def __init__(self, normalize: bool = True):
        """Initialize accuracy metric.

        Args:
            normalize: Whether to return fraction (True) or count (False)
        """
        super().__init__("accuracy", higher_is_better=True)
        self.normalize = normalize

    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> float:
        """Compute accuracy score.

        Args:
            predictions: Predicted labels
            references: True labels
            **kwargs: Additional parameters

        Returns:
            Accuracy score
        """
        if not predictions or not references:
            return 0.0

        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} vs {len(references)}"
            )

        correct = sum(1 for p, r in zip(predictions, references) if p == r)

        if self.normalize:
            return correct / len(predictions)
        else:
            return float(correct)


class Precision(BaseMetric):
    """Precision metric for classification."""

    def __init__(
        self, average: str = "binary", pos_label: Any = 1, zero_division: float = 0.0
    ):
        """Initialize precision metric.

        Args:
            average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
            pos_label: Positive class for binary classification
            zero_division: Value to return when there's a zero division
        """
        super().__init__(f"precision_{average}", higher_is_better=True)
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division

    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> float:
        """Compute precision score.

        Args:
            predictions: Predicted labels
            references: True labels
            **kwargs: Additional parameters

        Returns:
            Precision score
        """
        if not predictions or not references:
            return 0.0

        if HAS_SKLEARN:
            try:
                return sklearn_metrics.precision_score(
                    references,
                    predictions,
                    average=self.average if self.average != "binary" else None,
                    pos_label=self.pos_label if self.average == "binary" else None,
                    zero_division=self.zero_division,
                )
            except Exception as e:
                logger.warning(
                    f"sklearn precision failed: {e}, using basic implementation"
                )

        # Basic implementation
        if self.average == "binary":
            true_positives = sum(
                1
                for p, r in zip(predictions, references)
                if p == self.pos_label and r == self.pos_label
            )
            predicted_positives = sum(1 for p in predictions if p == self.pos_label)

            if predicted_positives == 0:
                return self.zero_division

            return true_positives / predicted_positives

        else:
            # Multi-class precision
            classes = set(references) | set(predictions)
            precisions = []
            weights = []

            for cls in classes:
                tp = sum(
                    1 for p, r in zip(predictions, references) if p == cls and r == cls
                )
                fp = sum(
                    1 for p, r in zip(predictions, references) if p == cls and r != cls
                )

                if tp + fp == 0:
                    precision = self.zero_division
                else:
                    precision = tp / (tp + fp)

                precisions.append(precision)
                weights.append(sum(1 for r in references if r == cls))

            if self.average == "macro":
                return np.mean(precisions)
            elif self.average == "weighted":
                total_weight = sum(weights)
                if total_weight == 0:
                    return 0.0
                return sum(p * w for p, w in zip(precisions, weights)) / total_weight
            elif self.average == "micro":
                # Micro-averaged precision is same as accuracy
                return sum(1 for p, r in zip(predictions, references) if p == r) / len(
                    predictions
                )

            return np.mean(precisions)


class Recall(BaseMetric):
    """Recall metric for classification."""

    def __init__(
        self, average: str = "binary", pos_label: Any = 1, zero_division: float = 0.0
    ):
        """Initialize recall metric.

        Args:
            average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
            pos_label: Positive class for binary classification
            zero_division: Value to return when there's a zero division
        """
        super().__init__(f"recall_{average}", higher_is_better=True)
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division

    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> float:
        """Compute recall score.

        Args:
            predictions: Predicted labels
            references: True labels
            **kwargs: Additional parameters

        Returns:
            Recall score
        """
        if not predictions or not references:
            return 0.0

        if HAS_SKLEARN:
            try:
                return sklearn_metrics.recall_score(
                    references,
                    predictions,
                    average=self.average if self.average != "binary" else None,
                    pos_label=self.pos_label if self.average == "binary" else None,
                    zero_division=self.zero_division,
                )
            except Exception as e:
                logger.warning(
                    f"sklearn recall failed: {e}, using basic implementation"
                )

        # Basic implementation
        if self.average == "binary":
            true_positives = sum(
                1
                for p, r in zip(predictions, references)
                if p == self.pos_label and r == self.pos_label
            )
            actual_positives = sum(1 for r in references if r == self.pos_label)

            if actual_positives == 0:
                return self.zero_division

            return true_positives / actual_positives

        else:
            # Multi-class recall
            classes = set(references) | set(predictions)
            recalls = []
            weights = []

            for cls in classes:
                tp = sum(
                    1 for p, r in zip(predictions, references) if p == cls and r == cls
                )
                fn = sum(
                    1 for p, r in zip(predictions, references) if p != cls and r == cls
                )

                if tp + fn == 0:
                    recall = self.zero_division
                else:
                    recall = tp / (tp + fn)

                recalls.append(recall)
                weights.append(sum(1 for r in references if r == cls))

            if self.average == "macro":
                return np.mean(recalls)
            elif self.average == "weighted":
                total_weight = sum(weights)
                if total_weight == 0:
                    return 0.0
                return sum(r * w for r, w in zip(recalls, weights)) / total_weight
            elif self.average == "micro":
                # Micro-averaged recall is same as accuracy
                return sum(1 for p, r in zip(predictions, references) if p == r) / len(
                    predictions
                )

            return np.mean(recalls)


class F1Score(BaseMetric):
    """F1 score metric for classification."""

    def __init__(
        self, average: str = "binary", pos_label: Any = 1, zero_division: float = 0.0
    ):
        """Initialize F1 score metric.

        Args:
            average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
            pos_label: Positive class for binary classification
            zero_division: Value to return when there's a zero division
        """
        super().__init__(f"f1_{average}", higher_is_better=True)
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division

        # Create precision and recall metrics
        self.precision_metric = Precision(average, pos_label, zero_division)
        self.recall_metric = Recall(average, pos_label, zero_division)

    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> float:
        """Compute F1 score.

        Args:
            predictions: Predicted labels
            references: True labels
            **kwargs: Additional parameters

        Returns:
            F1 score
        """
        if not predictions or not references:
            return 0.0

        if HAS_SKLEARN:
            try:
                return sklearn_metrics.f1_score(
                    references,
                    predictions,
                    average=self.average if self.average != "binary" else None,
                    pos_label=self.pos_label if self.average == "binary" else None,
                    zero_division=self.zero_division,
                )
            except Exception as e:
                logger.warning(f"sklearn f1 failed: {e}, using basic implementation")

        # Compute using precision and recall
        precision = self.precision_metric.compute(predictions, references)
        recall = self.recall_metric.compute(predictions, references)

        if precision + recall == 0:
            return self.zero_division

        return 2 * (precision * recall) / (precision + recall)


class ConfusionMatrix(BaseMetric):
    """Confusion matrix for detailed classification analysis."""

    def __init__(self, labels: Optional[List[Any]] = None):
        """Initialize confusion matrix.

        Args:
            labels: List of labels to index the matrix
        """
        super().__init__("confusion_matrix", higher_is_better=False)
        self.labels = labels
        self._matrix = None

    def compute(
        self, predictions: List[Any], references: List[Any], **kwargs
    ) -> np.ndarray:
        """Compute confusion matrix.

        Args:
            predictions: Predicted labels
            references: True labels
            **kwargs: Additional parameters

        Returns:
            Confusion matrix as numpy array
        """
        if not predictions or not references:
            return np.array([[]])

        if HAS_SKLEARN:
            try:
                self._matrix = sklearn_metrics.confusion_matrix(
                    references, predictions, labels=self.labels
                )
                return self._matrix
            except Exception as e:
                logger.warning(
                    f"sklearn confusion_matrix failed: {e}, using basic implementation"
                )

        # Basic implementation
        if self.labels is None:
            self.labels = sorted(set(references) | set(predictions))

        label_to_idx = {label: i for i, label in enumerate(self.labels)}
        n_labels = len(self.labels)

        matrix = np.zeros((n_labels, n_labels), dtype=int)

        for pred, ref in zip(predictions, references):
            if pred in label_to_idx and ref in label_to_idx:
                matrix[label_to_idx[ref], label_to_idx[pred]] += 1

        self._matrix = matrix
        return matrix

    def get_metrics(self) -> Dict[str, Any]:
        """Get derived metrics from confusion matrix.

        Returns:
            Dictionary of metrics
        """
        if self._matrix is None:
            return {}

        metrics = {}
        n_classes = len(self._matrix)

        # Per-class metrics
        for i, label in enumerate(self.labels or range(n_classes)):
            tp = self._matrix[i, i]
            fp = self._matrix[:, i].sum() - tp
            fn = self._matrix[i, :].sum() - tp
            tn = self._matrix.sum() - tp - fp - fn

            metrics[f"{label}_tp"] = int(tp)
            metrics[f"{label}_fp"] = int(fp)
            metrics[f"{label}_fn"] = int(fn)
            metrics[f"{label}_tn"] = int(tn)

            # Precision, recall, F1 for this class
            if tp + fp > 0:
                metrics[f"{label}_precision"] = tp / (tp + fp)
            else:
                metrics[f"{label}_precision"] = 0.0

            if tp + fn > 0:
                metrics[f"{label}_recall"] = tp / (tp + fn)
            else:
                metrics[f"{label}_recall"] = 0.0

            if metrics[f"{label}_precision"] + metrics[f"{label}_recall"] > 0:
                metrics[f"{label}_f1"] = (
                    2
                    * metrics[f"{label}_precision"]
                    * metrics[f"{label}_recall"]
                    / (metrics[f"{label}_precision"] + metrics[f"{label}_recall"])
                )
            else:
                metrics[f"{label}_f1"] = 0.0

        # Overall accuracy
        metrics["accuracy"] = np.diag(self._matrix).sum() / self._matrix.sum()

        return metrics


class ClassificationReport:
    """Generate comprehensive classification report."""

    def __init__(self, labels: Optional[List[Any]] = None):
        """Initialize classification report.

        Args:
            labels: List of labels
        """
        self.labels = labels
        self.metrics = {
            "accuracy": Accuracy(),
            "precision_macro": Precision(average="macro"),
            "recall_macro": Recall(average="macro"),
            "f1_macro": F1Score(average="macro"),
            "precision_weighted": Precision(average="weighted"),
            "recall_weighted": Recall(average="weighted"),
            "f1_weighted": F1Score(average="weighted"),
        }

        if labels and len(labels) == 2:
            # Add binary metrics
            self.metrics["precision_binary"] = Precision(
                average="binary", pos_label=labels[1]
            )
            self.metrics["recall_binary"] = Recall(
                average="binary", pos_label=labels[1]
            )
            self.metrics["f1_binary"] = F1Score(average="binary", pos_label=labels[1])

    def compute(
        self, predictions: List[Any], references: List[Any]
    ) -> Dict[str, float]:
        """Compute all classification metrics.

        Args:
            predictions: Predicted labels
            references: True labels

        Returns:
            Dictionary of metric scores
        """
        results = {}

        for name, metric in self.metrics.items():
            try:
                score = metric.compute(predictions, references)
                results[name] = score
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                results[name] = 0.0

        return results

    def format_report(self, predictions: List[Any], references: List[Any]) -> str:
        """Generate formatted classification report.

        Args:
            predictions: Predicted labels
            references: True labels

        Returns:
            Formatted report string
        """
        results = self.compute(predictions, references)

        # Build report
        lines = ["Classification Report", "=" * 50]

        # Overall metrics
        lines.append("\nOverall Metrics:")
        lines.append(f"  Accuracy:           {results.get('accuracy', 0):.4f}")
        lines.append(f"  Macro F1:           {results.get('f1_macro', 0):.4f}")
        lines.append(f"  Weighted F1:        {results.get('f1_weighted', 0):.4f}")

        # Detailed metrics
        lines.append("\nDetailed Metrics:")
        lines.append(f"  Precision (macro):  {results.get('precision_macro', 0):.4f}")
        lines.append(f"  Recall (macro):     {results.get('recall_macro', 0):.4f}")
        lines.append(
            f"  Precision (weighted): {results.get('precision_weighted', 0):.4f}"
        )
        lines.append(f"  Recall (weighted):  {results.get('recall_weighted', 0):.4f}")

        if "f1_binary" in results:
            lines.append("\nBinary Metrics:")
            lines.append(
                f"  Precision:          {results.get('precision_binary', 0):.4f}"
            )
            lines.append(f"  Recall:             {results.get('recall_binary', 0):.4f}")
            lines.append(f"  F1:                 {results.get('f1_binary', 0):.4f}")

        lines.append("=" * 50)

        return "\n".join(lines)


# Convenience instances
accuracy = Accuracy()
precision_macro = Precision(average="macro")
precision_micro = Precision(average="micro")
precision_weighted = Precision(average="weighted")
recall_macro = Recall(average="macro")
recall_micro = Recall(average="micro")
recall_weighted = Recall(average="weighted")
f1_macro = F1Score(average="macro")
f1_micro = F1Score(average="micro")
f1_weighted = F1Score(average="weighted")
