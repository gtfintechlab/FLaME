"""BenchForge - A professional benchmark engine for language models."""

# Version
__version__ = "0.4.0"

# Core Engine - Tasks
from bench_forge.tasks.base import BaseTask
from bench_forge.tasks.config import TaskConfig, PromptFormat
from bench_forge.tasks.registry import TaskRegistry, get_registry, register_task, task

# Core Engine - LLM
from bench_forge.llm.client import LLMClient
from bench_forge.llm.config import LLMConfig
from bench_forge.llm.batch import (
    BatchProcessor,
    BatchConfig,
    chunk_list,
    process_batch_with_retry,
)

# Core Engine - Inference
from bench_forge.engine.inference import InferenceEngine, InferenceResult

# Evaluation Engine
from bench_forge.engine.evaluation import EvaluationEngine, EvaluationResult

# Prompt Management
from bench_forge.prompts.templates import PromptTemplate
from bench_forge.prompts.registry import (
    PromptRegistry,
    get_prompt_registry,
    register_prompt,
    prompt,
)

# Response Extraction
from bench_forge.prompts.extractor import (
    ResponseExtractor,
    ExtractionStrategy,
    ExtractionResult,
    get_extractor,
)

# Metrics System
from bench_forge.metrics.base import BaseMetric
from bench_forge.metrics.wrappers import (
    ClassificationMetrics,
    accuracy_score,
    precision_recall_f1,
    confusion_matrix,
    TextMetrics,
    rouge_scores,
    bleu_score,
    text_similarity,
)

# Data Management
from bench_forge.data.loader import (
    LoaderConfig,
    DatasetLoader,
    HuggingFaceLoader,
    loader_factory,
    load_dataset,
)
from bench_forge.data.processor import (
    ProcessorConfig,
    DataProcessor,
    TextProcessor,
    process_dataset,
)
from bench_forge.data.splitter import SplitConfig, DataSplitter, train_test_split
from bench_forge.data.cache import CacheConfig, CacheManager, ResponseCache

# Configuration & Utilities
from bench_forge.utils.config import (
    BenchForgeConfig,
    ConfigManager,
    get_config,
    set_config,
    reload_config,
)
from bench_forge.utils.logging import setup_logging, get_logger, ColoredFormatter
from bench_forge.utils.validation import (
    ValidationError,
    ValidationResult,
    InputValidator,
    OutputValidator,
    validate_dataset,
    validate_prompt,
)
from bench_forge.utils.parallel import (
    ParallelConfig,
    ParallelExecutor,
    AsyncExecutor,
    parallel_map,
    async_gather,
)

# FLAME Integration
from bench_forge.flame.adapter import (
    FLAMEAdapter,
    FLAMETask,
    FLAMEConfig,
    flame_task,
)
from bench_forge.flame.utils import (
    args_to_config,
    load_flame_dataset,
    process_flame_results,
)

# Configure default logging
setup_logging(level="INFO")

__all__ = [
    # Version
    "__version__",
    # Core Engine - Tasks
    "BaseTask",
    "TaskConfig",
    "PromptFormat",
    "TaskRegistry",
    "get_registry",
    "register_task",
    "task",
    # Core Engine - LLM
    "LLMClient",
    "LLMConfig",
    "BatchProcessor",
    "BatchConfig",
    "chunk_list",
    "process_batch_with_retry",
    # Core Engine - Inference
    "InferenceEngine",
    "InferenceResult",
    # FLAME Integration
    "EvaluationEngine",
    "EvaluationResult",
    "ResponseExtractor",
    "ExtractionStrategy",
    "ExtractionResult",
    "get_extractor",
    # Prompt Management
    "PromptTemplate",
    "PromptRegistry",
    "get_prompt_registry",
    "register_prompt",
    "prompt",
    # Metrics System
    "BaseMetric",
    "ClassificationMetrics",
    "accuracy_score",
    "precision_recall_f1",
    "confusion_matrix",
    "TextMetrics",
    "rouge_scores",
    "bleu_score",
    "text_similarity",
    # Data Management
    "LoaderConfig",
    "DatasetLoader",
    "HuggingFaceLoader",
    "loader_factory",
    "load_dataset",
    "ProcessorConfig",
    "DataProcessor",
    "TextProcessor",
    "process_dataset",
    "SplitConfig",
    "DataSplitter",
    "train_test_split",
    "CacheConfig",
    "CacheManager",
    "ResponseCache",
    # Configuration & Utilities
    "BenchForgeConfig",
    "ConfigManager",
    "get_config",
    "set_config",
    "reload_config",
    "setup_logging",
    "get_logger",
    "ColoredFormatter",
    "ValidationError",
    "ValidationResult",
    "InputValidator",
    "OutputValidator",
    "validate_dataset",
    "validate_prompt",
    "ParallelConfig",
    "ParallelExecutor",
    "AsyncExecutor",
    "parallel_map",
    "async_gather",
    # FLAME Integration
    "FLAMEAdapter",
    "FLAMETask",
    "FLAMEConfig",
    "flame_task",
    "args_to_config",
    "load_flame_dataset",
    "process_flame_results",
]
