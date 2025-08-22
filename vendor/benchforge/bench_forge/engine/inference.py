"""Inference engine with logging, progress tracking, and result management."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import hashlib

from bench_forge.tasks.base import BaseTask
from bench_forge.tasks.registry import get_registry
from bench_forge.tasks.config import TaskConfig, PromptFormat
from bench_forge.llm.client import LLMClient
from bench_forge.llm.config import LLMConfig
from bench_forge.llm.batch import BatchProcessor


logger = logging.getLogger(__name__)


class InferenceResult:
    """Container for inference results with metadata."""

    def __init__(
        self,
        task_name: str,
        results_df: pd.DataFrame,
        metadata: Dict[str, Any],
        output_path: Optional[Path] = None,
    ):
        """Initialize inference result.

        Args:
            task_name: Name of the task
            results_df: DataFrame with results
            metadata: Result metadata
            output_path: Path where results were saved
        """
        self.task_name = task_name
        self.results_df = results_df
        self.metadata = metadata
        self.output_path = output_path
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_name": self.task_name,
            "num_samples": len(self.results_df),
            "timestamp": self.timestamp.isoformat(),
            "output_path": str(self.output_path) if self.output_path else None,
            "metadata": self.metadata,
        }

    def save(self, path: Path, format: str = "csv"):
        """Save results to file.

        Args:
            path: Output path
            format: Output format (csv, json, parquet)
        """
        if format == "csv":
            self.results_df.to_csv(path, index=False)
        elif format == "json":
            self.results_df.to_json(path, orient="records", indent=2)
        elif format == "parquet":
            self.results_df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved results to {path}")


class InferenceEngine:
    """Engine for running inference on benchmark tasks."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        batch_processor: Optional[BatchProcessor] = None,
        output_dir: Optional[Path] = None,
        save_results: bool = True,
        output_format: str = "csv",
    ):
        """Initialize inference engine.

        Args:
            llm_client: LLM client for completions
            batch_processor: Batch processor for parallel execution
            output_dir: Directory for saving results
            save_results: Whether to save results automatically
            output_format: Output format (csv, json, parquet)
        """
        self.llm_client = llm_client
        self.batch_processor = batch_processor or BatchProcessor()
        self.output_dir = Path(output_dir or "outputs")
        self.save_results = save_results
        self.output_format = output_format

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track run statistics
        self.stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_samples": 0,
            "total_time": 0.0,
        }

        logger.info(f"Initialized InferenceEngine with output_dir={self.output_dir}")

    def _generate_run_id(self, task_name: str) -> str:
        """Generate unique run ID.

        Args:
            task_name: Task name

        Returns:
            Unique run ID
        """
        timestamp = datetime.now().isoformat()
        content = f"{task_name}:{timestamp}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash_suffix}"

    def _create_metadata(
        self, task: BaseTask, llm_config: Optional[LLMConfig] = None, run_id: str = ""
    ) -> Dict[str, Any]:
        """Create metadata for the run.

        Args:
            task: Task instance
            llm_config: LLM configuration
            run_id: Run identifier

        Returns:
            Metadata dictionary
        """
        metadata = {
            "run_id": run_id,
            "task_name": task.name,
            "timestamp": datetime.now().isoformat(),
            "task_config": task.config.to_dict() if task.config else {},
        }

        if llm_config:
            metadata["llm_config"] = {
                "model": llm_config.model,
                "provider": llm_config.provider,
                "temperature": llm_config.temperature,
                "max_tokens": llm_config.max_tokens,
            }
        elif self.llm_client:
            metadata["llm_config"] = {
                "model": self.llm_client.config.model,
                "provider": self.llm_client.config.provider,
                "temperature": self.llm_client.config.temperature,
                "max_tokens": self.llm_client.config.max_tokens,
            }

        return metadata

    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif hasattr(obj, "value"):
            # For enums
            return obj.value
        elif hasattr(obj, "__dict__"):
            # For objects with __dict__ attribute
            return str(obj)
        else:
            return obj

    def run(
        self,
        task: Union[str, BaseTask],
        config: Optional[TaskConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        prompt_format: Optional[PromptFormat] = None,
        num_samples: Optional[int] = None,
        parallel: bool = False,
    ) -> InferenceResult:
        """Run inference for a task.

        Args:
            task: Task name or instance
            config: Task configuration (overrides defaults)
            llm_config: LLM configuration (overrides client defaults)
            prompt_format: Prompt format to use
            num_samples: Number of samples to process
            parallel: Whether to process batches in parallel

        Returns:
            InferenceResult with results and metadata

        Raises:
            ValueError: If task not found or configuration invalid
            RuntimeError: If inference fails
        """
        start_time = datetime.now()
        self.stats["total_runs"] += 1

        # Get task instance
        if isinstance(task, str):
            logger.info(f"Loading task: {task}")
            registry = get_registry()
            task_instance = registry.create_task(task, config)
        else:
            task_instance = task
            if config:
                task_instance.config = config

        # Validate configuration
        task_instance.validate_config()

        # Override num_samples if specified
        if num_samples is not None:
            task_instance.config.num_samples = num_samples

        # Generate run ID
        run_id = self._generate_run_id(task_instance.name)
        logger.info(f"Starting inference run: {run_id}")

        try:
            # Load dataset
            logger.info("Loading dataset...")
            dataset = task_instance.load_dataset()

            # Prepare prompts
            logger.info("Preparing prompts...")
            prompts = task_instance.prepare_prompts(dataset, prompt_format)
            self.stats["total_samples"] += len(prompts)

            # Get LLM client
            if llm_config:
                llm_client = LLMClient(llm_config)
            elif self.llm_client:
                llm_client = self.llm_client
            else:
                raise RuntimeError("No LLM client configured")

            # Process prompts
            logger.info(f"Processing {len(prompts)} prompts...")

            def process_batch(batch_prompts: List[Dict[str, Any]]) -> List[str]:
                """Process a batch of prompts."""
                prompt_texts = [p["prompt"] for p in batch_prompts]
                return llm_client.complete_batch(prompt_texts, show_progress=False)

            # Use batch processor
            responses = self.batch_processor.process(
                prompts,
                process_batch,
                parallel=parallel,
                progress_callback=lambda done, total: logger.info(
                    f"Progress: {done}/{total} batches"
                ),
            )

            # Process responses
            logger.info("Processing responses...")
            results_df = task_instance.process_responses(responses, prompts)

            # Add metadata columns
            results_df["run_id"] = run_id
            results_df["task"] = task_instance.name
            results_df["model"] = llm_client.config.model
            results_df["timestamp"] = datetime.now().isoformat()

            # Create metadata
            metadata = self._create_metadata(task_instance, llm_config, run_id)
            metadata["num_samples"] = len(results_df)
            metadata["duration_seconds"] = (datetime.now() - start_time).total_seconds()

            # Save results if configured
            output_path = None
            if self.save_results:
                output_path = self.output_dir / f"{run_id}.{self.output_format}"
                result = InferenceResult(
                    task_instance.name, results_df, metadata, output_path
                )
                result.save(output_path, self.output_format)

                # Save metadata (convert non-serializable objects to strings)
                metadata_path = self.output_dir / f"{run_id}_metadata.json"
                serializable_metadata = self._make_serializable(metadata)
                with open(metadata_path, "w") as f:
                    json.dump(serializable_metadata, f, indent=2, default=str)
                logger.info(f"Saved metadata to {metadata_path}")
            else:
                result = InferenceResult(task_instance.name, results_df, metadata)

            # Update statistics
            self.stats["successful_runs"] += 1
            self.stats["total_time"] += metadata["duration_seconds"]

            logger.info(f"Inference completed successfully: {run_id}")
            return result

        except Exception as e:
            self.stats["failed_runs"] += 1
            logger.error(f"Inference failed for {task_instance.name}: {e}")
            raise RuntimeError(f"Inference failed: {e}") from e

    def run_multiple(
        self,
        tasks: List[Union[str, BaseTask]],
        configs: Optional[Dict[str, TaskConfig]] = None,
        **kwargs,
    ) -> Dict[str, InferenceResult]:
        """Run inference for multiple tasks.

        Args:
            tasks: List of task names or instances
            configs: Task-specific configurations
            **kwargs: Additional arguments for run()

        Returns:
            Dictionary mapping task names to results
        """
        results = {}
        configs = configs or {}

        for task in tasks:
            task_name = task if isinstance(task, str) else task.name

            try:
                logger.info(f"Running task: {task_name}")
                config = configs.get(task_name)
                result = self.run(task, config=config, **kwargs)
                results[task_name] = result

            except Exception as e:
                logger.error(f"Failed to run task {task_name}: {e}")
                results[task_name] = None

        logger.info(f"Completed {len(results)} tasks")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()

        # Calculate success rate
        if stats["total_runs"] > 0:
            stats["success_rate"] = stats["successful_runs"] / stats["total_runs"]
            stats["avg_time_per_run"] = stats["total_time"] / stats["total_runs"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_time_per_run"] = 0.0

        # Add batch processor stats
        if self.batch_processor:
            stats["batch_stats"] = self.batch_processor.get_stats()

        # Add LLM client stats
        if self.llm_client:
            stats["llm_stats"] = self.llm_client.get_stats()

        return stats
