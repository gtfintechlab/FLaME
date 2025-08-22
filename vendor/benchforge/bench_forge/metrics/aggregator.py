"""Metric aggregation and management."""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

from bench_forge.metrics.base import BaseMetric, MetricResult
from bench_forge.metrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
)


logger = logging.getLogger(__name__)


class MetricAggregator:
    """Aggregate and manage multiple metrics."""

    def __init__(
        self,
        metrics: Optional[List[Union[BaseMetric, str]]] = None,
        task_type: str = "classification",
    ):
        """Initialize metric aggregator.

        Args:
            metrics: List of metrics or metric names
            task_type: Type of task (classification, generation, etc.)
        """
        self.task_type = task_type
        self.metrics = {}
        self.results_history = []

        # Add metrics
        if metrics:
            for metric in metrics:
                self.add_metric(metric)

        # Statistics
        self.stats = {
            "total_computations": 0,
            "total_samples_processed": 0,
            "last_computation": None,
        }

        logger.info(
            f"Initialized MetricAggregator for {task_type} with {len(self.metrics)} metrics"
        )

    def add_metric(self, metric: Union[BaseMetric, str]):
        """Add a metric to the aggregator.

        Args:
            metric: Metric instance or name
        """
        if isinstance(metric, str):
            metric = self._create_metric_from_name(metric)

        if not isinstance(metric, BaseMetric):
            raise TypeError(f"Expected BaseMetric, got {type(metric)}")

        self.metrics[metric.name] = metric
        logger.debug(f"Added metric: {metric.name}")

    def _create_metric_from_name(self, name: str) -> BaseMetric:
        """Create metric instance from name.

        Args:
            name: Metric name

        Returns:
            Metric instance

        Raises:
            ValueError: If metric name not recognized
        """
        name_lower = name.lower()

        # Classification metrics
        if name_lower == "accuracy":
            return Accuracy()
        elif name_lower == "precision" or name_lower == "precision_macro":
            return Precision(average="macro")
        elif name_lower == "precision_micro":
            return Precision(average="micro")
        elif name_lower == "precision_weighted":
            return Precision(average="weighted")
        elif name_lower == "recall" or name_lower == "recall_macro":
            return Recall(average="macro")
        elif name_lower == "recall_micro":
            return Recall(average="micro")
        elif name_lower == "recall_weighted":
            return Recall(average="weighted")
        elif name_lower == "f1" or name_lower == "f1_macro":
            return F1Score(average="macro")
        elif name_lower == "f1_micro":
            return F1Score(average="micro")
        elif name_lower == "f1_weighted":
            return F1Score(average="weighted")
        else:
            raise ValueError(f"Unknown metric: {name}")

    def compute(
        self,
        predictions: List[Any],
        references: List[Any],
        return_all: bool = True,
        **kwargs,
    ) -> Union[Dict[str, float], float]:
        """Compute all metrics.

        Args:
            predictions: Model predictions
            references: Ground truth values
            return_all: Return all metrics (True) or average (False)
            **kwargs: Additional parameters for metrics

        Returns:
            Dictionary of metric scores or average score
        """
        if not predictions or not references:
            logger.warning("Empty predictions or references")
            if return_all:
                return {name: 0.0 for name in self.metrics}
            else:
                return 0.0

        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
            )

        results = {}

        for name, metric in self.metrics.items():
            try:
                score = metric.compute(predictions, references, **kwargs)
                results[name] = score
                logger.debug(f"Computed {name}: {score:.4f}")
            except Exception as e:
                logger.error(f"Failed to compute {name}: {e}")
                results[name] = 0.0

        # Update statistics
        self.stats["total_computations"] += 1
        self.stats["total_samples_processed"] += len(predictions)
        self.stats["last_computation"] = datetime.now()

        # Store results
        result_record = {
            "timestamp": datetime.now(),
            "num_samples": len(predictions),
            "metrics": results.copy(),
        }
        self.results_history.append(result_record)

        if return_all:
            return results
        else:
            # Return average of all metrics
            return np.mean(list(results.values()))

    def compute_from_dataframe(
        self,
        df: pd.DataFrame,
        prediction_col: str = "prediction",
        reference_col: str = "reference",
        group_by: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict[str, float], pd.DataFrame]:
        """Compute metrics from DataFrame.

        Args:
            df: DataFrame with predictions and references
            prediction_col: Name of prediction column
            reference_col: Name of reference column
            group_by: Optional column to group by
            **kwargs: Additional parameters

        Returns:
            Metrics dictionary or DataFrame with grouped metrics
        """
        if group_by:
            # Compute metrics for each group
            results = []

            for group_value, group_df in df.groupby(group_by):
                predictions = group_df[prediction_col].tolist()
                references = group_df[reference_col].tolist()

                group_metrics = self.compute(predictions, references, **kwargs)
                group_metrics[group_by] = group_value
                results.append(group_metrics)

            return pd.DataFrame(results)
        else:
            # Compute overall metrics
            predictions = df[prediction_col].tolist()
            references = df[reference_col].tolist()

            return self.compute(predictions, references, **kwargs)

    def add_samples(
        self,
        predictions: List[Any],
        references: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add samples to all metrics for incremental computation.

        Args:
            predictions: Model predictions
            references: Ground truth values
            metadata: Optional metadata for samples
        """
        for metric in self.metrics.values():
            metric.add_batch(predictions, references, metadata)

        logger.debug(f"Added {len(predictions)} samples to all metrics")

    def aggregate_all(self) -> Dict[str, MetricResult]:
        """Aggregate all accumulated samples.

        Returns:
            Dictionary of MetricResult objects
        """
        results = {}

        for name, metric in self.metrics.items():
            try:
                result = metric.aggregate()
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to aggregate {name}: {e}")
                results[name] = MetricResult(name, 0.0, 0)

        return results

    def reset_all(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

        logger.debug("Reset all metrics")

    def get_summary(self) -> pd.DataFrame:
        """Get summary of all computed metrics.

        Returns:
            DataFrame with metric summaries
        """
        if not self.results_history:
            return pd.DataFrame()

        # Collect all metric values
        metric_data = {}

        for result in self.results_history:
            for metric_name, value in result["metrics"].items():
                if metric_name not in metric_data:
                    metric_data[metric_name] = []
                metric_data[metric_name].append(value)

        # Compute statistics
        summary_data = []

        for metric_name, values in metric_data.items():
            summary_data.append(
                {
                    "metric": metric_name,
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                }
            )

        return pd.DataFrame(summary_data)

    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metric values from history.

        Returns:
            Dictionary of best metric values
        """
        if not self.results_history:
            return {}

        best_metrics = {}

        for result in self.results_history:
            for metric_name, value in result["metrics"].items():
                if metric_name not in best_metrics:
                    best_metrics[metric_name] = value
                else:
                    # Check if higher is better
                    if metric_name in self.metrics:
                        metric = self.metrics[metric_name]
                        if metric.higher_is_better:
                            best_metrics[metric_name] = max(
                                best_metrics[metric_name], value
                            )
                        else:
                            best_metrics[metric_name] = min(
                                best_metrics[metric_name], value
                            )
                    else:
                        # Default to higher is better
                        best_metrics[metric_name] = max(
                            best_metrics[metric_name], value
                        )

        return best_metrics

    def format_results(
        self, results: Dict[str, float], title: str = "Metrics Report"
    ) -> str:
        """Format results as a readable string.

        Args:
            results: Metric results
            title: Report title

        Returns:
            Formatted report string
        """
        lines = [title, "=" * len(title)]

        # Group metrics by type
        accuracy_metrics = []
        precision_metrics = []
        recall_metrics = []
        f1_metrics = []
        other_metrics = []

        for name, value in results.items():
            if "accuracy" in name:
                accuracy_metrics.append((name, value))
            elif "precision" in name:
                precision_metrics.append((name, value))
            elif "recall" in name:
                recall_metrics.append((name, value))
            elif "f1" in name:
                f1_metrics.append((name, value))
            else:
                other_metrics.append((name, value))

        # Format each group
        for group_name, group_metrics in [
            ("Accuracy", accuracy_metrics),
            ("Precision", precision_metrics),
            ("Recall", recall_metrics),
            ("F1 Score", f1_metrics),
            ("Other", other_metrics),
        ]:
            if group_metrics:
                lines.append(f"\n{group_name}:")
                for name, value in group_metrics:
                    lines.append(f"  {name:20s}: {value:.4f}")

        return "\n".join(lines)

    def save_results(
        self,
        path: str,
        results: Optional[Dict[str, float]] = None,
        format: str = "json",
    ):
        """Save results to file.

        Args:
            path: Output path
            results: Results to save (uses last if None)
            format: Output format (json, csv)
        """
        import json

        if results is None and self.results_history:
            results = self.results_history[-1]["metrics"]

        if not results:
            logger.warning("No results to save")
            return

        if format == "json":
            with open(path, "w") as f:
                json.dump(results, f, indent=2)
        elif format == "csv":
            df = pd.DataFrame([results])
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved results to {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        stats["num_metrics"] = len(self.metrics)
        stats["num_computations"] = len(self.results_history)

        if self.results_history:
            stats["avg_samples_per_computation"] = stats[
                "total_samples_processed"
            ] / len(self.results_history)

        return stats
