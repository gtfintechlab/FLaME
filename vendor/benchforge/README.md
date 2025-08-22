# Bench Forge

A flexible and extensible evaluation framework for language models, extracted from the FLAME benchmark suite.

## Overview

The Bench Forge provides a unified framework for evaluating language models across various tasks and benchmarks. It offers:

- **Pluggable task system** - Easy registration of new evaluation tasks
- **Consistent interfaces** - Standardized APIs for inference and evaluation
- **Flexible configuration** - YAML-based configuration with CLI overrides
- **Structured output** - Hierarchical result organization with reproducible naming
- **Batch processing** - Efficient batch inference with retry mechanisms
- **Comprehensive logging** - Component-based logging with configurable levels

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/bench-forge.git
cd bench-forge

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

### As a Package

```bash
pip install bench-forge
```

## Quick Start

### Running Inference

```bash
# Run inference on a single task
bench-forge --mode inference --tasks my_task --model "openai/gpt-3.5-turbo"

# Run with configuration file
bench-forge --config config.yaml --mode inference --tasks task1 task2

# Override configuration parameters
bench-forge --config config.yaml --mode inference --tasks my_task \
  --model "together_ai/meta-llama/Llama-2-7b" \
  --max_tokens 256 --temperature 0.1
```

### Running Evaluation

```bash
# Evaluate inference results
bench-forge --mode evaluation --tasks my_task \
  --file_name "results/my_task/provider/model__my_task__r01__20240115__abc123.csv"
```

### List Available Tasks

```bash
# List all tasks
bench-forge list-tasks

# List only inference tasks
bench-forge list-tasks --mode inference

# List only evaluation tasks
bench-forge list-tasks --mode evaluation
```

## Configuration

### YAML Configuration

Create a `config.yaml` file:

```yaml
# Model configuration
model: "together_ai/meta-llama/Llama-2-7b-chat"
max_tokens: 128
temperature: 0.0
top_p: 0.9
batch_size: 10

# Prompt configuration
prompt_format: "zero_shot"

# Tasks to run
tasks:
  - task1
  - task2

# Task-specific overrides
task_config:
  task1:
    max_tokens: 256
    temperature: 0.1
  task2:
    prompt_format: "few_shot"

# Logging configuration
log_level: "INFO"
log_dir: "./logs"

# Output directories
results_dir: "./results"
evaluation_dir: "./evaluations"
```

### Command-Line Arguments

All configuration parameters can be overridden via command-line:

```bash
bench-forge --config config.yaml \
  --mode inference \
  --tasks task1 task2 \
  --model "openai/gpt-4" \
  --max_tokens 512 \
  --temperature 0.5 \
  --batch_size 5 \
  --prompt_format few_shot
```

## Creating Tasks

### Basic Task Implementation

```python
from bench_forge.core.base_task import BaseTask
from bench_forge.core.registry import task_decorator
import pandas as pd

@task_decorator("my_task", mode="inference")
def my_task_inference(args):
    """Run inference for my task."""
    # Load data
    data = load_my_data()
    
    # Prepare prompts
    prompts = prepare_prompts(data)
    
    # Run inference using batch processing
    from bench_forge.utils.batch import process_batch_with_retry
    
    model_config = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    
    responses = process_batch_with_retry(
        model_config,
        prompts,
        batch_idx=0,
        total_batches=1
    )
    
    # Return results as DataFrame
    return pd.DataFrame({
        "input": data,
        "output": responses
    })

@task_decorator("my_task", mode="evaluation")
def my_task_evaluate(file_name, args):
    """Evaluate results for my task."""
    # Load predictions
    df = pd.read_csv(file_name)
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Return results and metrics
    return df, pd.DataFrame([metrics])
```

### Using the Base Task Class

```python
from bench_forge.core.base_task import BaseTask, TaskConfig
from bench_forge.core.registry import register_task

class MyTask(BaseTask):
    def __init__(self):
        config = TaskConfig(
            name="my_task",
            dataset="my_dataset",
            metrics=["accuracy", "f1"],
            batch_size=10
        )
        super().__init__(config)
    
    def load_dataset(self, split="test"):
        # Load your dataset
        return load_my_dataset(split)
    
    def prepare_prompts(self, dataset, format="zero_shot"):
        # Prepare prompts from dataset
        prompts = []
        for item in dataset:
            prompt = f"Question: {item['question']}\nAnswer:"
            prompts.append([{"role": "user", "content": prompt}])
        return prompts
    
    def run_inference(self, prompts, model_config):
        # Run inference using the harness utilities
        from bench_forge.utils.batch import BatchProcessor
        
        processor = BatchProcessor(batch_size=model_config["batch_size"])
        responses = processor.process_llm_batches(prompts, model_config)
        
        return pd.DataFrame({"responses": responses})
    
    def evaluate_results(self, predictions, ground_truth=None):
        # Evaluate predictions
        metrics = calculate_metrics(predictions)
        metrics_df = pd.DataFrame([metrics])
        return predictions, metrics_df

# Register the task
task = MyTask()
register_task(
    "my_task",
    inference_fn=task.execute_inference,
    evaluation_fn=task.execute_evaluation
)
```

## Output Structure

The harness generates structured output paths:

```
results/
├── task_name/
│   ├── provider_name/
│   │   ├── model_family/  (optional)
│   │   │   └── model-slug__task-name__r01__20240115__abc123.csv
│   │   └── model-slug__task-name__r01__20240115__def456.csv

evaluations/
├── task_name/
│   ├── provider_name/
│   │   ├── model_family/  (optional)
│   │   │   ├── model-slug__task-name__r01__20240115__abc123.csv
│   │   │   └── model-slug__task-name__r01__20240115__abc123_metrics.csv
```

## API Reference

### Core Components

- `TaskRegistry` - Central registry for task management
- `BaseTask` - Abstract base class for task implementation
- `HarnessExecutor` - Main execution engine
- `HarnessConfig` - Configuration management

### Utilities

- `BatchProcessor` - Batch processing with retry logic
- `OutputManager` - Structured output path generation
- `LogConfig` - Logging configuration

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black bench_forge
isort bench_forge

# Type checking
mypy bench_forge
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bench_forge

# Run specific test file
pytest tests/test_registry.py
```

## Migration from FLAME

To migrate FLAME tasks to the bench-forge:

1. Copy task implementation files to `tasks/` directory
2. Update imports to use `bench_forge` modules
3. Register tasks using the harness registry
4. Update configuration to use harness format

Example migration:

```python
# Before (FLAME)
from flame.utils.batch_utils import process_batch_with_retry
from flame.utils.output_utils import generate_output_path

# After (bench-forge)
from bench_forge.utils.batch import process_batch_with_retry
from bench_forge.utils.output import generate_output_path
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This harness was extracted from the FLAME (Financial Language Model Evaluation) benchmark suite and incorporates design patterns from:

- EleutherAI's lm-evaluation-harness
- MLFoundations' evalchemy
- The FLAME team and contributors