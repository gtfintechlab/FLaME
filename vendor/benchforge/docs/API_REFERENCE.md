# BenchForge API Reference

## Table of Contents
1. [Core Engine](#core-engine)
2. [Task System](#task-system)
3. [LLM Interface](#llm-interface)
4. [Prompt Management](#prompt-management)
5. [Metrics System](#metrics-system)
6. [Data Management](#data-management)
7. [FLAME Integration](#flame-integration)
8. [Utilities](#utilities)

---

## Core Engine

### InferenceEngine

Main orchestrator for running benchmark inference tasks.

```python
class InferenceEngine:
    def __init__(
        self,
        llm_client: LLMClient,
        output_dir: Path = Path("results"),
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
        max_retries: int = 3,
        timeout: int = 60
    )
    
    def run(
        self,
        task: Union[str, BaseTask],
        config: TaskConfig,
        dataset: Optional[Any] = None,
        save_results: bool = True,
        progress_bar: bool = True
    ) -> InferenceResult
    
    def run_batch(
        self,
        tasks: List[Union[str, BaseTask]],
        configs: List[TaskConfig],
        parallel: bool = False
    ) -> List[InferenceResult]
```

### EvaluationEngine

Comprehensive evaluation system with metrics computation.

```python
class EvaluationEngine:
    def __init__(
        self,
        output_dir: Path = Path("evaluations"),
        metrics_config: Optional[Dict] = None
    )
    
    def evaluate(
        self,
        results_df: pd.DataFrame,
        task: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        save_results: bool = True
    ) -> EvaluationResult
    
    def compare(
        self,
        results: List[EvaluationResult],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame
```

### InferenceResult

Container for inference results.

```python
@dataclass
class InferenceResult:
    task_name: str
    dataset: str
    model: str
    results_df: pd.DataFrame
    output_path: Optional[Path]
    metadata: Dict[str, Any]
    statistics: Dict[str, float]
    errors: List[Dict[str, Any]]
```

### EvaluationResult

Container for evaluation results.

```python
@dataclass
class EvaluationResult:
    task_name: str
    dataset: str
    model: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray]
    per_class_metrics: Optional[Dict[str, Dict[str, float]]]
    statistics: Dict[str, Any]
    output_path: Optional[Path]
```

---

## Task System

### BaseTask

Abstract base class for all benchmark tasks.

```python
class BaseTask(ABC):
    def __init__(self, config: Optional[TaskConfig] = None):
        self.config = config or TaskConfig()
    
    @abstractmethod
    def create_prompt(
        self,
        sample: Dict[str, Any],
        format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt from sample."""
    
    def extract_response(
        self,
        response: str,
        strategy: Optional[ExtractionStrategy] = None
    ) -> Any:
        """Extract structured data from response."""
    
    def compute_metrics(
        self,
        predictions: List[Any],
        ground_truth: List[Any]
    ) -> Dict[str, float]:
        """Compute task-specific metrics."""
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate input sample."""
    
    def preprocess_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess sample before prompt creation."""
    
    def postprocess_response(self, response: str) -> str:
        """Postprocess model response."""
```

### TaskConfig

Configuration for benchmark tasks.

```python
@dataclass
class TaskConfig:
    name: str = "default_task"
    dataset: Optional[str] = None
    prompt_format: PromptFormat = PromptFormat.ZERO_SHOT
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.KEYWORD
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    max_samples: Optional[int] = None
    batch_size: int = 10
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # LLM parameters
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
```

### TaskRegistry

Global task registry with decorator support.

```python
class TaskRegistry:
    def register(self, name: str, task_class: Type[BaseTask]) -> None:
        """Register a task class."""
    
    def get(self, name: str) -> Type[BaseTask]:
        """Get task class by name."""
    
    def list_tasks(self) -> List[str]:
        """List all registered tasks."""
    
    def create_task(
        self,
        name: str,
        config: Optional[TaskConfig] = None
    ) -> BaseTask:
        """Create task instance."""

# Decorator for task registration
@task("task_name")
class MyTask(BaseTask):
    ...
```

---

## LLM Interface

### LLMClient

Unified interface for multiple LLM providers.

```python
class LLMClient:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None
    )
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Single completion."""
    
    def complete_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs
    ) -> List[str]:
        """Batch completion."""
    
    async def complete_async(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Async completion."""
    
    def stream(
        self,
        prompt: str,
        **kwargs
    ) -> Iterator[str]:
        """Streaming completion."""
```

### LLMConfig

Configuration for LLM providers.

```python
@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_retries: int = 3
    timeout: int = 60
    rate_limit: Optional[int] = None
    
    # Model parameters
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
```

### BatchProcessor

Efficient batch processing with retry logic.

```python
class BatchProcessor:
    def __init__(
        self,
        llm_client: LLMClient,
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0
    )
    
    def process(
        self,
        items: List[Any],
        prompt_fn: Callable[[Any], str],
        extract_fn: Optional[Callable[[str], Any]] = None,
        progress_bar: bool = True
    ) -> List[Any]:
        """Process items in batches."""
    
    async def process_async(
        self,
        items: List[Any],
        prompt_fn: Callable[[Any], str],
        extract_fn: Optional[Callable[[str], Any]] = None
    ) -> List[Any]:
        """Process items asynchronously."""
```

---

## Prompt Management

### PromptTemplate

Base template system for prompts.

```python
class PromptTemplate:
    def __init__(
        self,
        template: str,
        format: PromptFormat = PromptFormat.ZERO_SHOT,
        examples: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    )
    
    def format(
        self,
        **kwargs
    ) -> str:
        """Format template with variables."""
    
    def add_example(
        self,
        input: str,
        output: str,
        explanation: Optional[str] = None
    ) -> None:
        """Add example for few-shot learning."""
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to chat messages format."""
```

### ResponseExtractor

Multi-strategy response extraction system.

```python
class ResponseExtractor:
    def __init__(
        self,
        default_strategy: ExtractionStrategy = ExtractionStrategy.KEYWORD,
        case_sensitive: bool = False,
        strip_whitespace: bool = True
    )
    
    def extract(
        self,
        response: str,
        strategy: Optional[ExtractionStrategy] = None,
        **kwargs
    ) -> ExtractionResult:
        """Extract using specified strategy."""
    
    def extract_keyword(
        self,
        response: str,
        keywords: List[str],
        return_first: bool = True
    ) -> Optional[str]:
        """Extract based on keywords."""
    
    def extract_regex(
        self,
        response: str,
        pattern: str,
        group: int = 0
    ) -> Optional[str]:
        """Extract using regex."""
    
    def extract_json(
        self,
        response: str,
        key: Optional[str] = None
    ) -> Any:
        """Extract from JSON response."""
    
    def extract_with_confidence(
        self,
        response: str,
        candidates: List[str]
    ) -> Tuple[Optional[str], float]:
        """Extract with confidence score."""
```

### ExtractionStrategy

Available extraction strategies.

```python
class ExtractionStrategy(Enum):
    KEYWORD = "keyword"
    REGEX = "regex"
    JSON = "json"
    FIRST_LINE = "first_line"
    LAST_LINE = "last_line"
    FUNCTION = "function"
    CHAIN_OF_THOUGHT = "cot"
    STRUCTURED = "structured"
    FUZZY = "fuzzy"
    CONFIDENCE = "confidence"
```

---

## Metrics System

### ClassificationMetrics

Metrics for classification tasks.

```python
class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true: List, y_pred: List) -> float:
        """Compute accuracy."""
    
    @staticmethod
    def precision_recall_f1(
        y_true: List,
        y_pred: List,
        average: str = "macro"
    ) -> Dict[str, float]:
        """Compute precision, recall, and F1."""
    
    @staticmethod
    def confusion_matrix(
        y_true: List,
        y_pred: List,
        labels: Optional[List] = None
    ) -> np.ndarray:
        """Compute confusion matrix."""
    
    @staticmethod
    def per_class_metrics(
        y_true: List,
        y_pred: List,
        labels: Optional[List] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics."""
```

### TextMetrics

Metrics for text generation tasks.

```python
class TextMetrics:
    @staticmethod
    def rouge_scores(
        predictions: List[str],
        references: List[str],
        rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]
    ) -> Dict[str, float]:
        """Compute ROUGE scores."""
    
    @staticmethod
    def bleu_score(
        predictions: List[str],
        references: List[str],
        n_gram: int = 4
    ) -> float:
        """Compute BLEU score."""
    
    @staticmethod
    def text_similarity(
        text1: str,
        text2: str,
        method: str = "cosine"
    ) -> float:
        """Compute text similarity."""
```

---

## Data Management

### DatasetLoader

Multi-format dataset loading.

```python
class DatasetLoader:
    def __init__(
        self,
        config: Optional[LoaderConfig] = None,
        cache_dir: Optional[Path] = None
    )
    
    def load(
        self,
        source: str,
        format: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Dataset]:
        """Load dataset from source."""
    
    def load_huggingface(
        self,
        dataset_name: str,
        split: str = "test",
        **kwargs
    ) -> Dataset:
        """Load from HuggingFace."""
    
    def load_csv(
        self,
        path: Path,
        **kwargs
    ) -> pd.DataFrame:
        """Load from CSV."""
    
    def load_json(
        self,
        path: Path,
        **kwargs
    ) -> pd.DataFrame:
        """Load from JSON."""
```

### DataProcessor

Data preprocessing pipeline.

```python
class DataProcessor:
    def __init__(
        self,
        config: Optional[ProcessorConfig] = None
    )
    
    def process(
        self,
        data: Union[pd.DataFrame, Dataset],
        operations: List[str]
    ) -> Union[pd.DataFrame, Dataset]:
        """Apply processing operations."""
    
    def clean_text(
        self,
        text: str,
        remove_html: bool = True,
        remove_urls: bool = True,
        lowercase: bool = False
    ) -> str:
        """Clean text data."""
    
    def tokenize(
        self,
        text: str,
        tokenizer: Optional[Any] = None
    ) -> List[str]:
        """Tokenize text."""
    
    def normalize(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = "standard"
    ) -> pd.DataFrame:
        """Normalize numerical data."""
```

### CacheManager

Response and dataset caching.

```python
class CacheManager:
    def __init__(
        self,
        cache_dir: Path = Path(".cache"),
        ttl: int = 86400,  # 24 hours
        max_size: int = 1000
    )
    
    def get(
        self,
        key: str
    ) -> Optional[Any]:
        """Get cached value."""
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set cached value."""
    
    def has(self, key: str) -> bool:
        """Check if key exists."""
    
    def clear(self) -> None:
        """Clear all cache."""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
```

---

## FLAME Integration

### FLAMEAdapter

Main orchestration for FLAME workflows.

```python
class FLAMEAdapter:
    def __init__(
        self,
        registry: Optional[TaskRegistry] = None
    )
    
    def register_task(
        self,
        name: str,
        task_class: Type[FLAMETask]
    ) -> None:
        """Register FLAME task."""
    
    def create_task(
        self,
        name: str,
        config: Optional[FLAMEConfig] = None
    ) -> FLAMETask:
        """Create FLAME task instance."""
    
    def list_tasks(self) -> List[str]:
        """List available FLAME tasks."""
    
    def run_inference(
        self,
        task_name: str,
        llm_client: LLMClient,
        config: FLAMEConfig
    ) -> InferenceResult:
        """Run FLAME inference."""
    
    def run_evaluation(
        self,
        results: InferenceResult,
        metrics: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Run FLAME evaluation."""
```

### FLAMETask

Extended base task for financial benchmarks.

```python
class FLAMETask(BaseTask):
    def __init__(
        self,
        config: Optional[FLAMEConfig] = None
    )
    
    def load_dataset(
        self,
        split: str = "test"
    ) -> Dataset:
        """Load FLAME dataset."""
    
    def get_default_examples(self) -> List[Dict[str, Any]]:
        """Get default few-shot examples."""
    
    def compute_task_metrics(
        self,
        results_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute FLAME-specific metrics."""
```

### FLAMEConfig

FLAME-specific configuration.

```python
@dataclass
class FLAMEConfig(TaskConfig):
    huggingface_dataset: Optional[str] = None
    text_field: str = "text"
    label_field: str = "label"
    valid_labels: List[str] = field(default_factory=list)
    financial_domain: Optional[str] = None
    regulatory_compliance: bool = False
```

### flame_task Decorator

Decorator for FLAME task registration.

```python
def flame_task(name: str):
    """Decorator to register FLAME tasks."""
    def decorator(cls):
        cls._flame_task_name = name
        # Auto-register with adapter
        return cls
    return decorator

# Usage
@flame_task("fomc")
class FOMCTask(FLAMETask):
    ...
```

---

## Utilities

### ConfigManager

Global configuration management.

```python
class ConfigManager:
    def __init__(
        self,
        config_file: Optional[Path] = None
    )
    
    def load(
        self,
        path: Path
    ) -> BenchForgeConfig:
        """Load configuration from file."""
    
    def save(
        self,
        config: BenchForgeConfig,
        path: Path
    ) -> None:
        """Save configuration to file."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
    
    def update(self, **kwargs) -> None:
        """Update multiple values."""
```

### Logging

Structured logging with color support.

```python
def setup_logging(
    level: str = "INFO",
    format: Optional[str] = None,
    colored: bool = True,
    log_file: Optional[Path] = None
) -> None:
    """Setup global logging configuration."""

def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""

class ColoredFormatter(logging.Formatter):
    """Formatter with color support for terminal output."""
```

### Validation

Input and output validation utilities.

```python
class InputValidator:
    def validate(
        self,
        data: Any,
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate input data."""
    
    def validate_prompt(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        required_fields: Optional[List[str]] = None
    ) -> ValidationResult:
        """Validate prompt."""
    
    def validate_dataset(
        self,
        dataset: Union[pd.DataFrame, Dataset],
        required_columns: Optional[List[str]] = None,
        min_samples: int = 1
    ) -> ValidationResult:
        """Validate dataset."""

class OutputValidator:
    def validate(
        self,
        output: Any,
        expected_format: Any
    ) -> ValidationResult:
        """Validate output."""
    
    def validate_response(
        self,
        response: str,
        valid_values: Optional[List[str]] = None,
        min_length: int = 1
    ) -> ValidationResult:
        """Validate model response."""
```

### ParallelExecutor

Parallel task execution.

```python
class ParallelExecutor:
    def __init__(
        self,
        max_workers: int = 4,
        thread_safe: bool = True
    )
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        progress_bar: bool = True
    ) -> List[Any]:
        """Parallel map operation."""
    
    def execute(
        self,
        tasks: List[Callable],
        args_list: List[Tuple]
    ) -> List[Any]:
        """Execute tasks in parallel."""
    
    async def map_async(
        self,
        func: Callable,
        items: List[Any]
    ) -> List[Any]:
        """Async parallel map."""
```

---

## Common Usage Patterns

### Basic Inference
```python
from bench_forge import InferenceEngine, LLMClient, TaskConfig

# Setup
llm = LLMClient(provider="openai", model="gpt-4")
engine = InferenceEngine(llm_client=llm)

# Configure task
config = TaskConfig(
    name="sentiment",
    prompt_format=PromptFormat.ZERO_SHOT,
    batch_size=20
)

# Run
result = engine.run("sentiment", config)
print(f"Accuracy: {result.statistics['accuracy']}")
```

### Custom Task
```python
from bench_forge import task, BaseTask, TaskConfig

@task("custom_classification")
class CustomTask(BaseTask):
    def create_prompt(self, sample, format=None):
        return f"Classify: {sample['text']}"
    
    def extract_response(self, response):
        # Custom extraction logic
        return response.strip().upper()

# Use the task
engine.run("custom_classification", TaskConfig())
```

### FLAME Integration
```python
from flame.benchforge import (
    create_inference_engine,
    FLAMEConfig
)

# Simple FLAME workflow
engine = create_inference_engine()
config = FLAMEConfig(
    name="fomc",
    huggingface_dataset="fomc_minutes",
    valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"]
)

result = engine.run("fomc", config)
```

### Batch Processing
```python
from bench_forge import BatchProcessor, LLMClient

# Setup batch processor
llm = LLMClient()
processor = BatchProcessor(llm, batch_size=10)

# Process items
items = ["text1", "text2", "text3", ...]
results = processor.process(
    items,
    prompt_fn=lambda x: f"Analyze: {x}",
    extract_fn=lambda r: r.split("\n")[0]
)
```

### Custom Metrics
```python
from bench_forge.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def compute(self, y_true, y_pred):
        # Custom metric logic
        return {"custom_score": 0.95}

# Use in evaluation
engine = EvaluationEngine()
result = engine.evaluate(
    results_df,
    metrics=[CustomMetric()]
)
```

---

## Error Handling

All BenchForge components follow consistent error handling:

```python
from bench_forge.utils import ValidationError

try:
    result = engine.run(task, config)
except ValidationError as e:
    print(f"Validation failed: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
```

## Best Practices

1. **Always use configuration objects** instead of loose parameters
2. **Enable caching** for development and testing
3. **Use batch processing** for better throughput
4. **Implement proper error handling** with retries
5. **Monitor token usage** to control costs
6. **Validate inputs** before processing
7. **Use appropriate extraction strategies** for each task
8. **Log at appropriate levels** for debugging

## Environment Variables

```bash
# LLM Provider Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TOGETHER_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=hf_...

# Configuration
BENCHFORGE_CONFIG_PATH=/path/to/config.yaml
BENCHFORGE_CACHE_DIR=/path/to/cache
BENCHFORGE_OUTPUT_DIR=/path/to/outputs
BENCHFORGE_LOG_LEVEL=INFO

# Performance
BENCHFORGE_MAX_WORKERS=4
BENCHFORGE_BATCH_SIZE=10
BENCHFORGE_TIMEOUT=60
```

## Version Compatibility

- Python: 3.8+
- LiteLLM: 1.0+
- Pandas: 1.3+
- NumPy: 1.20+
- HuggingFace Datasets: 2.0+

## License

MIT License - See [LICENSE](../LICENSE.txt) for details.