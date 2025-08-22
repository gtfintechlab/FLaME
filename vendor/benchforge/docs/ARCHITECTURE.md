# BenchForge Architecture Documentation

## Overview

BenchForge is a professional benchmark engine for language models, providing a comprehensive infrastructure for running benchmarks, evaluations, and integrations with frameworks like FLAME.

## System Architecture

### Core Components

BenchForge is organized into several key subsystems:

#### 1. Core Engine (`engine/`)
- **InferenceEngine**: Orchestrates benchmark execution with LLM providers
- **EvaluationEngine**: Comprehensive evaluation with metrics aggregation
- **Features**:
  - Multi-provider LLM support via LiteLLM
  - Batch processing with retry logic
  - Result caching and persistence
  - Statistical analysis and error tracking

#### 2. Task System (`tasks/`)
- **BaseTask**: Abstract base class for all benchmark tasks
- **TaskConfig**: Configuration management for tasks
- **TaskRegistry**: Global task registration and discovery
- **Features**:
  - Decorator-based task registration
  - Flexible prompt formats (zero-shot, few-shot, chain-of-thought)
  - Task-specific metrics and validation

#### 3. LLM Interface (`llm/`)
- **LLMClient**: Unified interface for multiple LLM providers
- **LLMConfig**: Provider-agnostic configuration
- **BatchProcessor**: Efficient batch processing with retries
- **Features**:
  - Support for OpenAI, Anthropic, Together.ai, HuggingFace
  - Automatic rate limiting and retry logic
  - Streaming and async support
  - Token counting and cost tracking

#### 4. Prompt Management (`prompts/`)
- **PromptTemplate**: Base template system
- **ResponseExtractor**: Multi-strategy response extraction
- **PromptRegistry**: Global prompt template registry
- **Extraction Strategies**:
  - KEYWORD: Keyword-based extraction
  - REGEX: Regular expression patterns
  - JSON: Structured JSON parsing
  - CHAIN_OF_THOUGHT: CoT response parsing
  - FUZZY: Fuzzy matching for labels
  - CONFIDENCE: Extraction with confidence scores
  - And more...

#### 5. Metrics System (`metrics/`)
- **BaseMetric**: Abstract metric interface
- **ClassificationMetrics**: Accuracy, precision, recall, F1
- **TextMetrics**: ROUGE, BLEU, similarity scores
- **MetricsAggregator**: Cross-task metric aggregation
- **Features**:
  - Confusion matrix generation
  - Per-class metrics computation
  - Statistical significance testing
  - Custom metric support

#### 6. Data Management (`data/`)
- **DatasetLoader**: Multi-format dataset loading
- **DataProcessor**: Data preprocessing pipeline
- **DataSplitter**: Train/test/validation splitting
- **CacheManager**: Response and dataset caching
- **Supported Formats**:
  - HuggingFace datasets
  - CSV/JSON/JSONL files
  - Custom loaders

#### 7. Output Management (`output/`)
- **OutputManager**: Result serialization and storage
- **OutputVisualizer**: Metrics visualization
- **Features**:
  - Multiple output formats (JSON, CSV, Parquet)
  - Automatic result versioning
  - Metadata preservation
  - Visualization generation

#### 8. Utilities (`utils/`)
- **ConfigManager**: Global configuration management
- **Logging**: Colored, structured logging
- **Validation**: Input/output validation
- **ParallelExecutor**: Parallel task execution
- **Features**:
  - Thread-safe operations
  - Comprehensive error handling
  - Performance monitoring

### FLAME Integration (`flame/`)

BenchForge provides first-class support for FLAME (Financial Language Model Evaluation):

- **FLAMEAdapter**: Main orchestration for FLAME workflows
- **FLAMETask**: Extended base task for financial benchmarks
- **FLAMEConfig**: FLAME-specific configuration
- **Features**:
  - Financial domain awareness
  - HuggingFace dataset integration
  - Backward compatibility helpers
  - Task migration utilities

## Design Principles

### 1. Modularity
Each component is self-contained with clear interfaces, allowing for:
- Independent testing and development
- Easy extension and customization
- Component reuse across different contexts

### 2. Professional Standards
- Comprehensive error handling with recovery strategies
- Extensive logging at multiple levels
- Full type annotations for static analysis
- Thread-safe implementations where needed
- Production-ready with monitoring hooks

### 3. Extensibility
- Plugin architecture for new LLM providers
- Custom task registration via decorators
- Flexible metric system for domain-specific evaluations
- Strategy pattern for response extraction

### 4. Performance
- Batch processing for efficiency
- Response caching to minimize API calls
- Parallel execution support
- Memory-efficient data streaming

## Integration Points

### LLM Providers
BenchForge integrates with multiple LLM providers through LiteLLM:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Together.ai (Open models)
- HuggingFace (Local and hosted models)
- Custom providers via adapter pattern

### Data Sources
- HuggingFace Hub for datasets
- Local file systems (CSV, JSON, JSONL)
- Custom data loaders
- API-based data sources

### Output Destinations
- Local filesystem
- Cloud storage (S3, GCS)
- Database systems
- Visualization platforms

## Usage Patterns

### As a Library
```python
from bench_forge import InferenceEngine, LLMClient, TaskConfig

# Create components
llm = LLMClient(provider="openai", model="gpt-4")
engine = InferenceEngine(llm_client=llm)

# Run inference
result = engine.run(task="sentiment", config=TaskConfig(...))
```

### With FLAME
```python
from flame.benchforge import create_inference_engine, FLAMEConfig

# Use FLAME's simplified interface
engine = create_inference_engine()
result = engine.run("fomc", FLAMEConfig(...))
```

### Task Registration
```python
from bench_forge import task, BaseTask

@task("my_task")
class MyTask(BaseTask):
    def create_prompt(self, sample):
        return f"Analyze: {sample['text']}"
```

## Performance Characteristics

### Benchmarks
- **Throughput**: 100-500 samples/minute (provider-dependent)
- **Batch Size**: Optimized at 10-20 for most providers
- **Cache Hit Rate**: 60-80% on repeated runs
- **Memory Usage**: <1GB for typical workloads

### Scalability
- Horizontal scaling via parallel executors
- Vertical scaling with batch size tuning
- Distributed processing support (future)

## Security Considerations

- API key management via environment variables
- No credential logging
- Input sanitization for prompts
- Output validation for safety
- Rate limiting protection

## Future Roadmap

### Near Term
- Multi-modal support (images, audio)
- Streaming inference
- Real-time evaluation dashboard
- Enhanced caching strategies

### Long Term
- Distributed processing
- AutoML for prompt optimization
- Cost optimization algorithms
- Cross-benchmark analysis tools

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## References

- [API Reference](./API_REFERENCE.md)
- [Quick Start Guide](./QUICK_START_GUIDE.md)
- [FLAME Integration Guide](./FLAME_INTEGRATION.md)
- [Examples](../examples/)