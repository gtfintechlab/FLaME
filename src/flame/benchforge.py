"""FLAME integration with BenchForge.

This module provides seamless integration between FLAME and BenchForge,
leveraging BenchForge's professional infrastructure for financial benchmarks.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add BenchForge to path if needed
benchforge_path = Path(__file__).parent.parent.parent / "benchforge"
if benchforge_path.exists() and str(benchforge_path) not in sys.path:
    sys.path.insert(0, str(benchforge_path))

# Import BenchForge components
try:
    from bench_forge import (
        # Core components
        InferenceEngine,
        InferenceResult,
        EvaluationEngine,
        EvaluationResult,
        # LLM components
        LLMClient,
        LLMConfig,
        BatchProcessor,
        BatchConfig,
        # Task components
        TaskConfig,
        PromptFormat,
        get_registry,
        # FLAME-specific
        FLAMEAdapter,
        FLAMETask,
        FLAMEConfig,
        flame_task,
        args_to_config,
        load_flame_dataset,
        process_flame_results,
        # Prompt components
        ExtractionStrategy,
        ResponseExtractor,
        # Utilities
        setup_logging,
        get_config,
        chunk_list,  # For backward compatibility
    )

    BENCHFORGE_AVAILABLE = True

except ImportError as e:
    logging.error(f"BenchForge import failed: {e}")
    logging.error("BenchForge is required. Ensure the submodule is initialized.")
    raise ImportError(f"Failed to import BenchForge components: {e}") from e

logger = logging.getLogger(__name__)


# Re-export everything FLAME needs from BenchForge
__all__ = [
    # Core
    "InferenceEngine",
    "InferenceResult",
    "EvaluationEngine",
    "EvaluationResult",
    # LLM
    "LLMClient",
    "LLMConfig",
    "BatchProcessor",
    "BatchConfig",
    # Tasks
    "TaskConfig",
    "PromptFormat",
    "get_registry",
    # FLAME-specific
    "FLAMEAdapter",
    "FLAMETask",
    "FLAMEConfig",
    "flame_task",
    "args_to_config",
    "load_flame_dataset",
    "process_flame_results",
    # Prompt components
    "ExtractionStrategy",
    "ResponseExtractor",
    # Utilities
    "setup_logging",
    "get_config",
    "chunk_list",
    # Helper functions
    "create_llm_client",
    "create_inference_engine",
    "create_evaluation_engine",
    "run_flame_inference",
    "run_flame_evaluation",
    # Status
    "BENCHFORGE_AVAILABLE",
]


def create_llm_client(args=None, config: Optional[LLMConfig] = None) -> "LLMClient":
    """Create LLM client for FLAME.

    Args:
        args: Command-line arguments
        config: Optional LLMConfig override

    Returns:
        Configured LLMClient
    """
    if not BENCHFORGE_AVAILABLE:
        raise ImportError(
            "BenchForge is required. Install with: pip install -e ./benchforge"
        )

    if config is None and args is not None:
        configs = args_to_config(args)
        config = configs["llm_config"]
    elif config is None:
        # Default configuration
        config = LLMConfig(
            provider="litellm",
            model="together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
            max_tokens=256,
            temperature=0.0,
        )

    return LLMClient(config)


def create_inference_engine(
    llm_client: Optional["LLMClient"] = None,
    output_dir: Optional[Path] = None,
    args=None,
) -> "InferenceEngine":
    """Create inference engine for FLAME.

    Args:
        llm_client: Optional LLM client
        output_dir: Output directory for results
        args: Optional command-line arguments

    Returns:
        Configured InferenceEngine
    """
    if not BENCHFORGE_AVAILABLE:
        raise ImportError(
            "BenchForge is required. Install with: pip install -e ./benchforge"
        )

    # Create LLM client if not provided
    if llm_client is None:
        llm_client = create_llm_client(args)

    # Set output directory
    if output_dir is None:
        output_dir = Path("results")

    return InferenceEngine(llm_client=llm_client, output_dir=output_dir)


def create_evaluation_engine(output_dir: Optional[Path] = None) -> "EvaluationEngine":
    """Create evaluation engine for FLAME.

    Args:
        output_dir: Output directory for evaluations

    Returns:
        Configured EvaluationEngine
    """
    if not BENCHFORGE_AVAILABLE:
        raise ImportError(
            "BenchForge is required. Install with: pip install -e ./benchforge"
        )

    if output_dir is None:
        output_dir = Path("evaluations")

    return EvaluationEngine(output_dir=output_dir)


def run_flame_inference(
    task_name: str,
    args=None,
    llm_client: Optional["LLMClient"] = None,
    config: Optional["FLAMEConfig"] = None,
    save_results: bool = True,
) -> "InferenceResult":
    """Run inference for a FLAME task.

    Args:
        task_name: Name of the task to run
        args: Command-line arguments
        llm_client: Optional LLM client
        config: Optional task configuration
        save_results: Whether to save results

    Returns:
        InferenceResult
    """
    if not BENCHFORGE_AVAILABLE:
        raise ImportError(
            "BenchForge is required. Install with: pip install -e ./benchforge"
        )

    # Create configurations if needed
    if config is None and args is not None:
        configs = args_to_config(args, task_name)
        config = configs["task_config"]
    elif config is None:
        config = FLAMEConfig(name=task_name)

    # Create LLM client if needed
    if llm_client is None:
        llm_client = create_llm_client(args)

    # Create inference engine
    engine = create_inference_engine(
        llm_client=llm_client, output_dir=config.results_dir
    )

    # Run inference
    logger.info(f"Running FLAME inference for task: {task_name}")
    result = engine.run(task=task_name, config=config, save_results=save_results)

    logger.info(f"Inference complete: {len(result.results_df)} samples processed")

    return result


def run_flame_evaluation(
    results_path: Union[str, Path],
    task_name: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    save_results: bool = True,
) -> "EvaluationResult":
    """Run evaluation for FLAME results.

    Args:
        results_path: Path to results file
        task_name: Optional task name
        metrics: Optional list of metrics to compute
        save_results: Whether to save evaluation results

    Returns:
        EvaluationResult
    """
    if not BENCHFORGE_AVAILABLE:
        raise ImportError(
            "BenchForge is required. Install with: pip install -e ./benchforge"
        )

    # Create evaluation engine
    engine = create_evaluation_engine()

    # Run evaluation
    logger.info(f"Running FLAME evaluation for: {results_path}")
    result = engine.evaluate(
        results_path=results_path,
        task=task_name,
        metrics=metrics,
        save_results=save_results,
    )

    logger.info(f"Evaluation complete: {result}")

    return result


# Backward compatibility functions for FLAME
def process_batch_with_retry(
    args,
    messages_batch: List[Any],
    batch_idx: int,
    total_batches: int,
    max_tokens: Optional[int] = None,
) -> List[Any]:
    """FLAME-compatible batch processing function.

    This maintains backward compatibility with FLAME's original interface
    while using BenchForge underneath.

    Args:
        args: Command-line arguments
        messages_batch: Batch of messages
        batch_idx: Current batch index
        total_batches: Total number of batches
        max_tokens: Optional max tokens override

    Returns:
        List of responses in FLAME format
    """
    if not BENCHFORGE_AVAILABLE:
        raise ImportError(
            "BenchForge is required. Install with: pip install -e ./benchforge"
        )

    # Use FLAME adapter for compatibility
    adapter = FLAMEAdapter()
    return adapter.process_batch_compatibility(
        args, messages_batch, batch_idx, total_batches
    )


# Convenience function to check BenchForge status
def check_benchforge_status() -> Dict[str, Any]:
    """Check BenchForge integration status.

    Returns:
        Status dictionary
    """
    status: Dict[str, Any] = {
        "available": BENCHFORGE_AVAILABLE,
        "version": None,
        "registered_tasks": [],
    }

    if BENCHFORGE_AVAILABLE:
        try:
            import bench_forge

            status["version"] = bench_forge.__version__

            registry = get_registry()
            status["registered_tasks"] = registry.list_tasks()
        except Exception as e:
            status["error"] = str(e)

    return status
