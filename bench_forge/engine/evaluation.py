"""Professional evaluation engine for benchmark results.

This module provides comprehensive evaluation capabilities with metrics aggregation,
statistical analysis, and result persistence.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

import pandas as pd
import numpy as np

from bench_forge.metrics.base import BaseMetric
from bench_forge.metrics.wrappers import ClassificationMetrics, TextMetrics
from bench_forge.tasks.registry import get_registry
from bench_forge.utils.validation import OutputValidator, ValidationError
from bench_forge.utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results container.

    Attributes:
        task_name: Name of the evaluated task
        dataset: Dataset identifier
        model: Model identifier
        metrics: Dictionary of computed metrics
        metadata: Additional evaluation metadata
        timestamp: Evaluation timestamp
        duration: Total evaluation duration in seconds
        num_samples: Number of samples evaluated
        num_errors: Number of evaluation errors
        error_details: Details of any errors encountered
        per_sample_scores: Optional per-sample metric scores
        confusion_matrix: Optional confusion matrix for classification
        statistical_summary: Statistical summary of results
    """

    task_name: str
    dataset: str
    model: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    num_samples: int = 0
    num_errors: int = 0
    error_details: List[Dict[str, Any]] = field(default_factory=list)
    per_sample_scores: Optional[pd.DataFrame] = None
    confusion_matrix: Optional[np.ndarray] = None
    statistical_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "task_name": self.task_name,
            "dataset": self.dataset,
            "model": self.model,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "num_samples": self.num_samples,
            "num_errors": self.num_errors,
            "error_details": self.error_details,
            "statistical_summary": self.statistical_summary,
        }

        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix.tolist()

        return result

    def __str__(self) -> str:
        """String representation of evaluation results."""
        lines = [
            f"Evaluation Results for {self.task_name}",
            f"Dataset: {self.dataset}",
            f"Model: {self.model}",
            f"Samples: {self.num_samples}",
            f"Duration: {self.duration:.2f}s",
            "\nMetrics:",
        ]

        for metric, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.4f}")
            else:
                lines.append(f"  {metric}: {value}")

        if self.num_errors > 0:
            lines.append(f"\nErrors: {self.num_errors}")

        return "\n".join(lines)


class EvaluationEngine:
    """Professional evaluation engine with comprehensive metrics support.

    This engine provides:
    - Multi-metric evaluation
    - Statistical analysis
    - Error handling and recovery
    - Result persistence
    - Visualization support
    - Comparison capabilities
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        metrics_registry: Optional[Dict[str, BaseMetric]] = None,
        validator: Optional[OutputValidator] = None,
        cache_results: bool = True,
        raise_on_error: bool = False,
        statistical_analysis: bool = True,
        save_per_sample: bool = False,
    ):
        """Initialize evaluation engine.

        Args:
            output_dir: Directory for saving evaluation results
            metrics_registry: Custom metrics registry
            validator: Output validator for results validation
            cache_results: Whether to cache evaluation results
            raise_on_error: Whether to raise exceptions on errors
            statistical_analysis: Whether to compute statistical summaries
            save_per_sample: Whether to save per-sample scores
        """
        self.output_dir = (
            Path(output_dir) if output_dir else Path(get_config().evaluation_dir)
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.metrics_registry = metrics_registry or self._initialize_default_metrics()

        # Initialize validator
        self.validator = validator or OutputValidator()

        # Configuration
        self.cache_results = cache_results
        self.raise_on_error = raise_on_error
        self.statistical_analysis = statistical_analysis
        self.save_per_sample = save_per_sample

        # State tracking
        self._cache = {}
        self._stats = defaultdict(int)

        logger.info(f"Initialized EvaluationEngine with output_dir: {self.output_dir}")

    def _initialize_default_metrics(self) -> Dict[str, BaseMetric]:
        """Initialize default metrics registry."""
        metrics = {}

        # Classification metrics
        classification = ClassificationMetrics()
        metrics["accuracy"] = classification
        metrics["precision"] = classification
        metrics["recall"] = classification
        metrics["f1"] = classification
        metrics["f1_macro"] = classification
        metrics["f1_micro"] = classification
        metrics["f1_weighted"] = classification

        # Text metrics
        text = TextMetrics()
        metrics["rouge"] = text
        metrics["rouge_l"] = text
        metrics["bleu"] = text
        metrics["similarity"] = text

        return metrics

    def evaluate(
        self,
        results_path: Optional[Union[str, Path]] = None,
        results_df: Optional[pd.DataFrame] = None,
        task: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        save_results: bool = True,
        output_format: str = "all",
        **kwargs,
    ) -> EvaluationResult:
        """Evaluate model results with comprehensive metrics.

        Args:
            results_path: Path to results CSV file
            results_df: Results DataFrame (alternative to results_path)
            task: Task name for task-specific evaluation
            metrics: List of metrics to compute
            save_results: Whether to save evaluation results
            output_format: Output format ('json', 'csv', 'all')
            **kwargs: Additional arguments for metrics

        Returns:
            EvaluationResult with computed metrics

        Raises:
            ValueError: If neither results_path nor results_df provided
            ValidationError: If results validation fails
        """
        start_time = datetime.now()

        # Load results
        if results_df is not None:
            df = results_df
            model = df.attrs.get("model", "unknown")
            dataset = df.attrs.get("dataset", task or "unknown")
        elif results_path:
            df = self._load_results(results_path)
            # Extract model and dataset from filename if possible
            path = Path(results_path)
            parts = path.stem.split("_")
            model = "_".join(parts[1:-3]) if len(parts) > 3 else "unknown"
            dataset = parts[0] if parts else task or "unknown"
        else:
            raise ValueError("Either results_path or results_df must be provided")

        # Validate results format
        validation_result = self.validator.validate_results(df)
        if not validation_result.is_valid:
            if self.raise_on_error:
                raise ValidationError(
                    f"Results validation failed: {validation_result.errors}"
                )
            else:
                logger.warning(
                    f"Results validation warnings: {validation_result.warnings}"
                )

        # Get task configuration if available
        task_config = None
        if task:
            try:
                registry = get_registry()
                task_instance = registry.create_task(task)
                task_config = task_instance.config
                if not metrics:
                    metrics = task_config.metrics
            except Exception as e:
                logger.warning(f"Could not load task configuration: {e}")

        # Default metrics if not specified
        if not metrics:
            metrics = self._infer_metrics(df)

        # Compute metrics
        metrics_results = {}
        error_details = []

        for metric_name in metrics:
            try:
                metric_value = self._compute_metric(
                    df, metric_name, task_config=task_config, **kwargs
                )
                metrics_results[metric_name] = metric_value
                logger.debug(f"Computed {metric_name}: {metric_value}")
            except Exception as e:
                error_msg = f"Failed to compute {metric_name}: {str(e)}"
                logger.error(error_msg)
                error_details.append(
                    {"metric": metric_name, "error": str(e), "type": type(e).__name__}
                )
                if self.raise_on_error:
                    raise

        # Compute confusion matrix for classification tasks
        confusion_matrix = None
        if self._is_classification_task(df):
            try:
                confusion_matrix = self._compute_confusion_matrix(df)
            except Exception as e:
                logger.warning(f"Could not compute confusion matrix: {e}")

        # Compute statistical summary
        statistical_summary = {}
        if self.statistical_analysis:
            statistical_summary = self._compute_statistical_summary(df, metrics_results)

        # Compute per-sample scores if requested
        per_sample_scores = None
        if self.save_per_sample:
            per_sample_scores = self._compute_per_sample_scores(df, metrics)

        # Create evaluation result
        duration = (datetime.now() - start_time).total_seconds()

        result = EvaluationResult(
            task_name=task or "unknown",
            dataset=dataset,
            model=model,
            metrics=metrics_results,
            metadata={
                "evaluation_engine_version": "0.4.0",
                "metrics_computed": metrics,
                "validation_passed": validation_result.is_valid,
                "output_format": output_format,
            },
            timestamp=start_time,
            duration=duration,
            num_samples=len(df),
            num_errors=len(error_details),
            error_details=error_details,
            per_sample_scores=per_sample_scores,
            confusion_matrix=confusion_matrix,
            statistical_summary=statistical_summary,
        )

        # Save results if requested
        if save_results:
            self._save_results(result, output_format)

        # Update statistics
        self._update_stats(result)

        # Cache if enabled
        if self.cache_results:
            cache_key = f"{task}_{model}_{dataset}"
            self._cache[cache_key] = result

        logger.info(f"Evaluation complete: {result}")

        return result

    def _load_results(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load results from file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix == ".json":
            return pd.read_json(path)
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            # Try to infer format
            return pd.read_csv(path)

    def _infer_metrics(self, df: pd.DataFrame) -> List[str]:
        """Infer appropriate metrics based on data."""
        metrics = []

        # Check for classification columns
        if all(col in df.columns for col in ["extracted_response", "ground_truth"]):
            metrics.extend(["accuracy", "f1_macro"])

        # Check for text generation columns
        if "raw_response" in df.columns:
            if "reference" in df.columns or "ground_truth" in df.columns:
                metrics.extend(["rouge_l", "bleu"])

        # Default fallback
        if not metrics:
            metrics = ["accuracy"]

        logger.info(f"Inferred metrics: {metrics}")
        return metrics

    def _compute_metric(
        self,
        df: pd.DataFrame,
        metric_name: str,
        task_config: Optional[Any] = None,
        **kwargs,
    ) -> float:
        """Compute a single metric."""
        # Get metric instance
        if metric_name in self.metrics_registry:
            metric = self.metrics_registry[metric_name]
        else:
            # Try to create metric dynamically
            if (
                "accuracy" in metric_name
                or "f1" in metric_name
                or "precision" in metric_name
                or "recall" in metric_name
            ):
                metric = ClassificationMetrics()
            elif "rouge" in metric_name or "bleu" in metric_name:
                metric = TextMetrics()
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

        # Prepare predictions and ground truth
        if "extracted_response" in df.columns:
            predictions = df["extracted_response"].tolist()
        elif "prediction" in df.columns:
            predictions = df["prediction"].tolist()
        elif "raw_response" in df.columns:
            predictions = df["raw_response"].tolist()
        else:
            raise ValueError("No prediction column found in results")

        if "ground_truth" in df.columns:
            ground_truth = df["ground_truth"].tolist()
        elif "label" in df.columns:
            ground_truth = df["label"].tolist()
        elif "reference" in df.columns:
            ground_truth = df["reference"].tolist()
        else:
            raise ValueError("No ground truth column found in results")

        # Compute metric
        if "accuracy" in metric_name:
            return metric.accuracy(predictions, ground_truth)
        elif "f1_macro" in metric_name:
            return metric.f1_score(predictions, ground_truth, average="macro")
        elif "f1_micro" in metric_name:
            return metric.f1_score(predictions, ground_truth, average="micro")
        elif "f1_weighted" in metric_name:
            return metric.f1_score(predictions, ground_truth, average="weighted")
        elif "precision" in metric_name:
            if "macro" in metric_name:
                return metric.precision(predictions, ground_truth, average="macro")
            elif "micro" in metric_name:
                return metric.precision(predictions, ground_truth, average="micro")
            else:
                return metric.precision(predictions, ground_truth, average="weighted")
        elif "recall" in metric_name:
            if "macro" in metric_name:
                return metric.recall(predictions, ground_truth, average="macro")
            elif "micro" in metric_name:
                return metric.recall(predictions, ground_truth, average="micro")
            else:
                return metric.recall(predictions, ground_truth, average="weighted")
        elif "rouge" in metric_name:
            if "rouge_l" in metric_name:
                scores = metric.rouge_scores(predictions, ground_truth)
                return scores.get("rouge-l", {}).get("f", 0.0)
            else:
                scores = metric.rouge_scores(predictions, ground_truth)
                return scores.get("rouge-1", {}).get("f", 0.0)
        elif "bleu" in metric_name:
            return metric.bleu_score(predictions, ground_truth)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    def _is_classification_task(self, df: pd.DataFrame) -> bool:
        """Check if this is a classification task."""
        if "extracted_response" not in df.columns:
            return False

        # Check if responses are categorical
        unique_values = df["extracted_response"].nunique()
        total_values = len(df)

        # Heuristic: if unique values < 20% of total, likely classification
        return unique_values < total_values * 0.2

    def _compute_confusion_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Compute confusion matrix for classification tasks."""
        metric = ClassificationMetrics()

        predictions = df["extracted_response"].tolist()
        ground_truth = df.get("ground_truth", df.get("label", [])).tolist()

        if not ground_truth:
            return None

        return metric.confusion_matrix(predictions, ground_truth)

    def _compute_statistical_summary(
        self, df: pd.DataFrame, metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compute statistical summary of results."""
        summary = {
            "total_samples": len(df),
            "metrics_summary": {},
        }

        # Summarize metrics
        if metrics:
            summary["metrics_summary"] = {
                "mean": np.mean(list(metrics.values())),
                "std": np.std(list(metrics.values())),
                "min": min(metrics.values()),
                "max": max(metrics.values()),
            }

        # Analyze errors if present
        if "error" in df.columns:
            error_df = df[df["error"].notna()]
            summary["error_analysis"] = {
                "total_errors": len(error_df),
                "error_rate": len(error_df) / len(df),
            }

        # Response length statistics
        if "raw_response" in df.columns:
            response_lengths = df["raw_response"].str.len()
            summary["response_stats"] = {
                "mean_length": response_lengths.mean(),
                "std_length": response_lengths.std(),
                "min_length": response_lengths.min(),
                "max_length": response_lengths.max(),
            }

        return summary

    def _compute_per_sample_scores(
        self, df: pd.DataFrame, metrics: List[str]
    ) -> pd.DataFrame:
        """Compute per-sample metric scores."""
        # This would compute metrics for each sample individually
        # For now, return None as this requires metric-specific implementation
        return None

    def _save_results(self, result: EvaluationResult, format: str):
        """Save evaluation results to disk."""
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = f"eval_{result.task_name}_{result.model}_{timestamp}"

        if format in ["json", "all"]:
            json_path = self.output_dir / f"{base_name}.json"
            with open(json_path, "w") as f:
                # Convert numpy types to native Python types for JSON serialization
                import numpy as np

                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(v) for v in obj]
                    else:
                        return obj

                json_data = convert_numpy(result.to_dict())
                json.dump(json_data, f, indent=2, default=str)
            logger.info(f"Saved JSON results to: {json_path}")

        if format in ["csv", "all"]:
            csv_path = self.output_dir / f"{base_name}_metrics.csv"
            metrics_df = pd.DataFrame([result.metrics])
            metrics_df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV metrics to: {csv_path}")

            # Save per-sample scores if available
            if result.per_sample_scores is not None:
                sample_path = self.output_dir / f"{base_name}_samples.csv"
                result.per_sample_scores.to_csv(sample_path, index=False)
                logger.info(f"Saved per-sample scores to: {sample_path}")

    def _update_stats(self, result: EvaluationResult):
        """Update internal statistics."""
        self._stats["total_evaluations"] += 1
        self._stats["total_samples"] += result.num_samples
        self._stats["total_errors"] += result.num_errors
        self._stats["total_duration"] += result.duration

        # Track metrics
        for metric_name, value in result.metrics.items():
            self._stats[f"metric_{metric_name}_sum"] += value
            self._stats[f"metric_{metric_name}_count"] += 1

    def compare_results(
        self,
        results: List[EvaluationResult],
        comparison_metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare multiple evaluation results.

        Args:
            results: List of evaluation results to compare
            comparison_metrics: Metrics to include in comparison

        Returns:
            DataFrame with comparison results
        """
        if not results:
            return pd.DataFrame()

        # Determine metrics to compare
        if not comparison_metrics:
            # Use all common metrics
            all_metrics = set()
            for r in results:
                all_metrics.update(r.metrics.keys())
            comparison_metrics = sorted(all_metrics)

        # Build comparison DataFrame
        data = []
        for result in results:
            row = {
                "task": result.task_name,
                "model": result.model,
                "dataset": result.dataset,
                "samples": result.num_samples,
            }

            for metric in comparison_metrics:
                row[metric] = result.metrics.get(metric, np.nan)

            data.append(row)

        comparison_df = pd.DataFrame(data)

        # Add ranking columns for each metric
        for metric in comparison_metrics:
            if metric in comparison_df.columns:
                comparison_df[f"{metric}_rank"] = comparison_df[metric].rank(
                    ascending=False, method="min"
                )

        return comparison_df

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation engine statistics."""
        stats = dict(self._stats)

        # Compute averages
        if stats.get("total_evaluations", 0) > 0:
            stats["avg_samples_per_eval"] = (
                stats["total_samples"] / stats["total_evaluations"]
            )
            stats["avg_duration"] = stats["total_duration"] / stats["total_evaluations"]
            stats["error_rate"] = stats["total_errors"] / stats["total_samples"]

        # Compute metric averages
        metric_avgs = {}
        for key in list(stats.keys()):
            if key.startswith("metric_") and key.endswith("_sum"):
                metric_name = key.replace("metric_", "").replace("_sum", "")
                count_key = f"metric_{metric_name}_count"
                if count_key in stats and stats[count_key] > 0:
                    metric_avgs[f"avg_{metric_name}"] = stats[key] / stats[count_key]

        stats.update(metric_avgs)

        return stats

    def clear_cache(self):
        """Clear the results cache."""
        self._cache.clear()
        logger.info("Cleared evaluation cache")
