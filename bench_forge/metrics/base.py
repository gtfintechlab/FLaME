"""Base metric interface and utilities."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging
from datetime import datetime
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for metric computation result."""

    name: str
    value: float
    count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "count": self.count,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(
        self,
        name: str,
        higher_is_better: bool = True,
        requires_probabilities: bool = False,
    ):
        """Initialize metric.

        Args:
            name: Metric name
            higher_is_better: Whether higher values are better
            requires_probabilities: Whether metric needs probability scores
        """
        self.name = name
        self.higher_is_better = higher_is_better
        self.requires_probabilities = requires_probabilities

        # Accumulator for incremental computation
        self._predictions = []
        self._references = []
        self._scores = []
        self._metadata = []

        # Statistics
        self.stats = {
            "total_computations": 0,
            "total_samples": 0,
            "last_computed": None,
        }

        logger.debug(f"Initialized metric: {name}")

    @abstractmethod
    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> float:
        """Compute metric score.

        Args:
            predictions: Model predictions
            references: Ground truth values
            **kwargs: Additional parameters

        Returns:
            Metric score
        """
        pass

    def add(
        self, prediction: Any, reference: Any, metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a single prediction-reference pair.

        Args:
            prediction: Single prediction
            reference: Single reference
            metadata: Optional metadata
        """
        self._predictions.append(prediction)
        self._references.append(reference)
        if metadata:
            self._metadata.append(metadata)

        logger.debug(f"Added sample to {self.name}: total={len(self._predictions)}")

    def add_batch(
        self,
        predictions: List[Any],
        references: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add multiple prediction-reference pairs.

        Args:
            predictions: List of predictions
            references: List of references
            metadata: Optional list of metadata
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length"
            )

        self._predictions.extend(predictions)
        self._references.extend(references)

        if metadata:
            if len(metadata) != len(predictions):
                raise ValueError("Metadata must have same length as predictions")
            self._metadata.extend(metadata)

        logger.debug(
            f"Added batch to {self.name}: batch_size={len(predictions)}, total={len(self._predictions)}"
        )

    def reset(self):
        """Reset accumulated data."""
        self._predictions.clear()
        self._references.clear()
        self._scores.clear()
        self._metadata.clear()
        logger.debug(f"Reset metric: {self.name}")

    def aggregate(self, **kwargs) -> MetricResult:
        """Compute metric on accumulated data.

        Args:
            **kwargs: Additional parameters

        Returns:
            MetricResult
        """
        if not self._predictions:
            logger.warning(f"No data to aggregate for {self.name}")
            return MetricResult(self.name, 0.0, 0)

        score = self.compute(self._predictions, self._references, **kwargs)

        # Update statistics
        self.stats["total_computations"] += 1
        self.stats["total_samples"] += len(self._predictions)
        self.stats["last_computed"] = datetime.now()

        result = MetricResult(
            name=self.name,
            value=score,
            count=len(self._predictions),
            metadata={
                "higher_is_better": self.higher_is_better,
                "requires_probabilities": self.requires_probabilities,
            },
        )

        # Store score
        self._scores.append(score)

        logger.info(f"Computed {self.name}: {score:.4f} (n={len(self._predictions)})")

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get metric statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()

        if self._scores:
            stats["mean_score"] = np.mean(self._scores)
            stats["std_score"] = np.std(self._scores)
            stats["min_score"] = np.min(self._scores)
            stats["max_score"] = np.max(self._scores)
            stats["num_scores"] = len(self._scores)

        stats["accumulated_samples"] = len(self._predictions)

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"

    def __str__(self) -> str:
        """String representation."""
        return self.name


class AveragedMetric(BaseMetric):
    """Base class for metrics that average over samples."""

    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> float:
        """Compute averaged metric.

        Args:
            predictions: Model predictions
            references: Ground truth values
            **kwargs: Additional parameters

        Returns:
            Averaged metric score
        """
        if not predictions:
            return 0.0

        scores = []
        for pred, ref in zip(predictions, references):
            score = self.compute_single(pred, ref, **kwargs)
            scores.append(score)

        return np.mean(scores)

    @abstractmethod
    def compute_single(self, prediction: Any, reference: Any, **kwargs) -> float:
        """Compute metric for a single sample.

        Args:
            prediction: Single prediction
            reference: Single reference
            **kwargs: Additional parameters

        Returns:
            Score for single sample
        """
        pass


class ThresholdMetric(BaseMetric):
    """Base class for metrics with configurable thresholds."""

    def __init__(
        self, name: str, threshold: float = 0.5, higher_is_better: bool = True, **kwargs
    ):
        """Initialize threshold metric.

        Args:
            name: Metric name
            threshold: Decision threshold
            higher_is_better: Whether higher values are better
            **kwargs: Additional parameters
        """
        super().__init__(name, higher_is_better, **kwargs)
        self.threshold = threshold

    def set_threshold(self, threshold: float):
        """Set decision threshold.

        Args:
            threshold: New threshold value
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        self.threshold = threshold
        logger.debug(f"Set threshold for {self.name}: {threshold}")

    def optimize_threshold(
        self,
        predictions: List[float],
        references: List[Any],
        thresholds: Optional[List[float]] = None,
    ) -> float:
        """Find optimal threshold.

        Args:
            predictions: Probability predictions
            references: Ground truth values
            thresholds: Thresholds to try (default: 0.1 to 0.9)

        Returns:
            Optimal threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)

        best_score = -np.inf if self.higher_is_better else np.inf
        best_threshold = 0.5

        for thresh in thresholds:
            self.threshold = thresh
            score = self.compute(predictions, references)

            if (self.higher_is_better and score > best_score) or (
                not self.higher_is_better and score < best_score
            ):
                best_score = score
                best_threshold = thresh

        self.threshold = best_threshold
        logger.info(
            f"Optimal threshold for {self.name}: {best_threshold:.2f} (score: {best_score:.4f})"
        )

        return best_threshold
