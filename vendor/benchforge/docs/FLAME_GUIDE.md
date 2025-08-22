# FLAME-Specific Integration Guide

## Overview

This guide provides FLAME-specific instructions for using BenchForge infrastructure. For general integration patterns, see the [Integration Guide](./INTEGRATION_GUIDE.md).

FLAME (Financial Language Model Evaluation) can leverage BenchForge's professional-grade infrastructure for running financial benchmark tasks.

## Quick Start

### Installation

```bash
# Install BenchForge
cd benchforge
pip install -e .

# Install FLAME with BenchForge support
cd ../
pip install -e .
```

### Basic Usage

```python
from flame.benchforge import create_inference_engine, FLAMEConfig

# Create engine
engine = create_inference_engine()

# Configure task
config = FLAMEConfig(
    name="fomc",
    dataset="fomc_minutes",
    prompt_format="zero_shot",
    batch_size=20
)

# Run inference
result = engine.run("fomc", config)
print(f"Processed {len(result.results_df)} samples")
```

## Task Migration

### Creating a FLAME Task

FLAME tasks extend BenchForge's task system with financial domain awareness:

```python
from flame.benchforge import flame_task, FLAMETask, FLAMEConfig
from flame.benchforge import PromptFormat, ExtractionStrategy

@flame_task("financial_sentiment")
class FinancialSentimentTask(FLAMETask):
    """Financial sentiment classification task."""
    
    def __init__(self, config=None):
        if config is None:
            config = FLAMEConfig(
                name="financial_sentiment",
                huggingface_dataset="financial_phrasebank",
                valid_labels=["POSITIVE", "NEGATIVE", "NEUTRAL"],
                financial_domain="sentiment_analysis",
                extraction_strategy=ExtractionStrategy.KEYWORD
            )
        super().__init__(config)
    
    def create_prompt(self, sample, format=None):
        """Create task-specific prompt."""
        format = format or self.config.prompt_format
        text = sample.get(self.config.text_field, "")
        
        if format == PromptFormat.ZERO_SHOT:
            return f"""Classify the financial sentiment:
            
Text: {text}

Sentiment (POSITIVE/NEGATIVE/NEUTRAL):"""
        
        elif format == PromptFormat.FEW_SHOT:
            examples = self.get_default_examples()
            prompt = "Examples:\n"
            for ex in examples:
                prompt += f"Text: {ex['text']}\n"
                prompt += f"Sentiment: {ex['label']}\n\n"
            prompt += f"Text: {text}\nSentiment:"
            return prompt
    
    def get_default_examples(self):
        """Provide few-shot examples."""
        return [
            {"text": "Profits exceeded expectations", "label": "POSITIVE"},
            {"text": "Revenue declined sharply", "label": "NEGATIVE"},
            {"text": "The meeting is scheduled for Tuesday", "label": "NEUTRAL"}
        ]
    
    def compute_task_metrics(self, results_df):
        """Compute financial-specific metrics."""
        metrics = super().compute_task_metrics(results_df)
        
        # Add domain-specific metrics
        if 'confidence' in results_df.columns:
            metrics['avg_confidence'] = results_df['confidence'].mean()
        
        return metrics
```

### Task Registration

Tasks are automatically registered when decorated with `@flame_task`:

```python
# In your tasks module
from flame.tasks import register_all_flame_tasks

# Register all available tasks
registered = register_all_flame_tasks()
print(f"Registered tasks: {registered}")
```

## Configuration

### FLAMEConfig

FLAME-specific configuration extends BenchForge's TaskConfig:

```python
from flame.benchforge import FLAMEConfig, PromptFormat

config = FLAMEConfig(
    # Basic configuration
    name="task_name",
    dataset="dataset_name",
    
    # FLAME-specific fields
    huggingface_dataset="hf_dataset_name",
    text_field="text",  # Column name for text
    label_field="label",  # Column name for labels
    valid_labels=["LABEL1", "LABEL2"],
    financial_domain="domain_type",
    regulatory_compliance=False,
    
    # Prompt configuration
    prompt_format=PromptFormat.ZERO_SHOT,
    extraction_strategy=ExtractionStrategy.KEYWORD,
    
    # Model parameters
    max_tokens=256,
    temperature=0.0,
    batch_size=20,
    
    # Evaluation metrics
    metrics=["accuracy", "f1_weighted"]
)
```

### Environment Configuration

```bash
# .env file
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=hf_...

# Optional configuration
FLAME_RESULTS_DIR=./results
FLAME_EVALUATION_DIR=./evaluations
BENCHFORGE_CACHE_DIR=./cache
```

## Response Extraction

BenchForge provides sophisticated extraction strategies for financial tasks:

```python
from flame.benchforge import ResponseExtractor, ExtractionStrategy

extractor = ResponseExtractor(
    default_strategy=ExtractionStrategy.KEYWORD,
    case_sensitive=False
)

# Extract with different strategies
response = "The sentiment is clearly POSITIVE based on the earnings report."

# Keyword extraction
result = extractor.extract(
    response,
    strategy=ExtractionStrategy.KEYWORD,
    keywords=["POSITIVE", "NEGATIVE", "NEUTRAL"]
)
print(f"Extracted: {result.value} (confidence: {result.confidence})")

# Fuzzy matching
result = extractor.extract(
    response,
    strategy=ExtractionStrategy.FUZZY,
    candidates=["POSITIVE", "NEGATIVE", "NEUTRAL"],
    threshold=0.8
)

# JSON extraction
json_response = '{"sentiment": "POSITIVE", "confidence": 0.95}'
result = extractor.extract(
    json_response,
    strategy=ExtractionStrategy.JSON,
    key="sentiment"
)
```

## Evaluation

### Running Evaluation

```python
from flame.benchforge import create_evaluation_engine

# Create evaluation engine
eval_engine = create_evaluation_engine()

# Evaluate results
evaluation = eval_engine.evaluate(
    results_path="results/fomc/results.csv",
    task="fomc",
    metrics=["accuracy", "f1_macro", "confusion_matrix"]
)

# Display results
print(f"Accuracy: {evaluation.metrics['accuracy']:.4f}")
print(f"F1 Macro: {evaluation.metrics['f1_macro']:.4f}")

# Per-class metrics
for label, metrics in evaluation.per_class_metrics.items():
    print(f"{label}: Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}")
```

### Custom Metrics

```python
from bench_forge.metrics import BaseMetric

class FinancialMetric(BaseMetric):
    """Custom metric for financial tasks."""
    
    def compute(self, y_true, y_pred, **kwargs):
        # Custom financial metric logic
        score = calculate_financial_score(y_true, y_pred)
        return {"financial_score": score}

# Use in evaluation
evaluation = eval_engine.evaluate(
    results_df,
    metrics=[FinancialMetric()]
)
```

## Command-Line Interface

### Using the FLAME CLI with BenchForge

```bash
# Run inference
python src/flame/main_benchforge.py \
    --mode inference \
    --task fomc \
    --model "together_ai/meta-llama/Llama-3-8B-Instruct" \
    --max_tokens 256 \
    --batch_size 20 \
    --prompt_format zero_shot

# Run evaluation
python src/flame/main_benchforge.py \
    --mode evaluate \
    --task fomc \
    --file_name results/fomc/results_20240815.csv \
    --metrics accuracy f1_macro confusion_matrix

# Check status
python src/flame/main_benchforge.py --mode status
```

## Dataset Integration

### Loading FLAME Datasets

```python
from flame.benchforge import load_flame_dataset

# Load from HuggingFace
dataset = load_flame_dataset(
    "financial_phrasebank",
    split="test",
    cache_dir="./cache"
)

# Load from local file
dataset = load_flame_dataset(
    "path/to/dataset.csv",
    format="csv"
)

# With preprocessing
dataset = load_flame_dataset(
    "fomc_minutes",
    preprocess_fn=lambda x: x.lower(),
    filter_fn=lambda x: len(x['text']) > 100
)
```

### Processing Results

```python
from flame.benchforge import process_flame_results

# Process and save results
output_path = process_flame_results(
    results_df,
    task_name="fomc",
    model_name="gpt-4",
    output_dir="./results",
    include_metadata=True
)

print(f"Results saved to: {output_path}")
```

## Backward Compatibility

BenchForge maintains backward compatibility with existing FLAME code:

```python
# Legacy FLAME functions still work
from flame.benchforge import chunk_list, process_batch_with_retry

# Chunk data for batch processing
chunks = list(chunk_list(data, batch_size=10))

# Process with retry (compatibility wrapper)
responses = process_batch_with_retry(
    args,
    messages,
    batch_start=0,
    batch_idx=0
)
```

## Advanced Features

### Parallel Processing

```python
from flame.benchforge import ParallelExecutor

# Create parallel executor
executor = ParallelExecutor(max_workers=4)

# Process multiple tasks in parallel
tasks = ["fomc", "fpb", "headlines"]
configs = [FLAMEConfig(name=t) for t in tasks]

results = executor.map(
    lambda t, c: engine.run(t, c),
    tasks,
    configs
)
```

### Caching

```python
from flame.benchforge import CacheManager

# Setup caching
cache = CacheManager(
    cache_dir="./cache",
    ttl=86400  # 24 hours
)

# Use cached results
cache_key = f"{task_name}_{model}_{dataset}"
if cache.has(cache_key):
    results = cache.get(cache_key)
else:
    results = engine.run(task_name, config)
    cache.set(cache_key, results)
```

### Custom Extractors

```python
from flame.benchforge import ResponseExtractor

class FinancialExtractor(ResponseExtractor):
    """Custom extractor for financial responses."""
    
    def extract_financial_entity(self, response):
        """Extract financial entities."""
        # Custom extraction logic
        entities = parse_financial_entities(response)
        return ExtractionResult(
            value=entities,
            strategy="financial_entity",
            confidence=0.95
        )

# Register custom extractor
task.extractor = FinancialExtractor()
```

## Migration Examples

### Migrating FOMC Task

Original FLAME code:
```python
# Old FLAME implementation
def fomc_inference(args):
    dataset = load_dataset("fomc")
    prompts = [create_fomc_prompt(x) for x in dataset]
    responses = call_llm_batch(prompts)
    return process_results(responses)
```

With BenchForge:
```python
# New BenchForge implementation
@flame_task("fomc")
class FOMCTask(FLAMETask):
    def __init__(self):
        super().__init__(FLAMEConfig(
            name="fomc",
            huggingface_dataset="fomc_minutes",
            valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"]
        ))
    
    def create_prompt(self, sample, format=None):
        return f"Classify: {sample['text']}"

# Usage
engine = create_inference_engine()
result = engine.run("fomc", FLAMEConfig())
```

### Migrating Multiple Tasks

```python
# Define task mappings
TASK_MIGRATIONS = {
    "fomc": FOMCTask,
    "fpb": FPBTask,
    "headlines": HeadlinesTask,
    "numclaim": NumClaimTask,
}

# Register all tasks
for name, task_class in TASK_MIGRATIONS.items():
    flame_task(name)(task_class)

# Run all tasks
for task_name in TASK_MIGRATIONS:
    result = engine.run(task_name, FLAMEConfig(name=task_name))
    print(f"{task_name}: {result.statistics['accuracy']}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**
```python
# Ensure BenchForge is installed
import sys
sys.path.append("./benchforge")
```

2. **Task Not Found**
```python
# Register tasks before use
from flame.tasks import register_all_flame_tasks
register_all_flame_tasks()
```

3. **API Key Issues**
```python
# Check environment variables
import os
assert "OPENAI_API_KEY" in os.environ
```

4. **Memory Issues**
```python
# Reduce batch size
config.batch_size = 5
```

### Debug Mode

```python
# Enable debug logging
from flame.benchforge import setup_logging
setup_logging(level="DEBUG", colored=True)

# Verbose inference
result = engine.run(
    task_name,
    config,
    progress_bar=True,
    verbose=True
)
```

## Performance Optimization

### Batch Size Tuning

```python
# Find optimal batch size
batch_sizes = [5, 10, 20, 50]
for batch_size in batch_sizes:
    config.batch_size = batch_size
    start = time.time()
    result = engine.run(task, config)
    elapsed = time.time() - start
    print(f"Batch {batch_size}: {elapsed:.2f}s")
```

### Token Optimization

```python
# Minimize token usage
config = FLAMEConfig(
    max_tokens=50,  # Reduce for classification
    temperature=0.0,  # Deterministic
    prompt_format=PromptFormat.ZERO_SHOT  # Fewer tokens than few-shot
)
```

### Parallel Execution

```python
# Use async processing
import asyncio

async def run_async():
    tasks = [
        engine.run_async("task1", config1),
        engine.run_async("task2", config2),
    ]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(run_async())
```

## Best Practices

1. **Task Design**
   - Keep prompts concise and clear
   - Use appropriate extraction strategies
   - Validate inputs and outputs
   - Implement task-specific metrics

2. **Configuration**
   - Use environment variables for API keys
   - Enable caching during development
   - Set appropriate batch sizes
   - Use deterministic settings for reproducibility

3. **Error Handling**
   - Implement retry logic for API failures
   - Validate dataset format before processing
   - Log errors with context
   - Save intermediate results

4. **Performance**
   - Batch requests when possible
   - Cache expensive operations
   - Use parallel processing for independent tasks
   - Monitor token usage

## Next Steps

- Review [Integration Guide](./INTEGRATION_GUIDE.md) for general patterns
- Read the [API Reference](./API_REFERENCE.md) for detailed documentation
- Check [Architecture](./ARCHITECTURE.md) for system design
- See [examples/](../examples/) for implementation patterns
