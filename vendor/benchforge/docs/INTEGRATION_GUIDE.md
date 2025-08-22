# BenchForge Integration Guide

## Overview

This guide covers how to integrate BenchForge with benchmark implementations and how benchmarks can use BenchForge infrastructure.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Integration Patterns](#integration-patterns)
3. [Migration Strategy](#migration-strategy)
4. [API Reference](#api-reference)
5. [Best Practices](#best-practices)

## Architecture Overview

BenchForge follows a clean separation of concerns:

```
┌─────────────────────────────────────────┐
│         Benchmark Layer (FLAME)         │
│  - Task implementations                 │
│  - Domain-specific logic                │
│  - Results management                   │
└────────────┬────────────────────────────┘
             │ imports/uses
┌────────────▼────────────────────────────┐
│     BenchForge Infrastructure Layer     │
│  - LLM Client abstraction               │
│  - Output management                    │
│  - Prompt engineering                   │
│  - Metrics calculation                  │
│  - Batch processing                     │
└─────────────────────────────────────────┘
```

### Core Design Principles

- 🏗️ **Pure Infrastructure**: BenchForge provides infrastructure components only
- 🔧 **No Benchmark Logic**: No benchmark-specific code in BenchForge
- 📦 **Clean API**: Simple, consistent API for benchmark implementations
- 🔄 **Optional Loading**: Can dynamically load external benchmarks

## Integration Patterns

### Pattern 1: Direct Import (Primary)

Benchmarks import BenchForge as a library or submodule:

```python
# In your benchmark's inference module
from bench_forge.engine import InferenceEngine
from bench_forge.llm.client import LLMClient
from bench_forge.output.manager import OutputManager

class MyBenchmarkInference:
    def __init__(self):
        self.engine = InferenceEngine()
        self.client = LLMClient(config)
        self.output = OutputManager(base_dir="results/")
    
    def run(self, dataset):
        # Use BenchForge infrastructure
        return self.engine.process(dataset, self.prompt_function)
```

### Pattern 2: Plugin Architecture (Secondary)

BenchForge can dynamically load benchmark plugins:

```python
# benchforge/plugins/loader.py
class BenchmarkPlugin:
    """Interface for loadable benchmarks"""
    
    @abstractmethod
    def get_tasks(self) -> Dict[str, Task]:
        pass
    
    @abstractmethod
    def get_config(self) -> Config:
        pass

# Your benchmark implements the plugin
class MyBenchmarkPlugin(BenchmarkPlugin):
    def get_tasks(self):
        return {
            "task1": Task1(),
            "task2": Task2(),
        }
```

## Migration Strategy

### For FLAME Integration

#### Step 1: Remove Benchmark Logic from BenchForge

```bash
# Move any benchmark-specific code out
mv benchforge/bench_forge/flame/tasks/* → ../FLAME/src/flame/benchforge_tasks/
rm -rf benchforge/benchforge_results/  # Results belong to benchmarks
```

#### Step 2: Create Clean BenchForge API

```python
# benchforge/bench_forge/api.py
class BenchForgeAPI:
    """Clean API for benchmark implementations"""
    
    def create_llm_client(self, config):
        return LLMClient(config)
    
    def create_output_manager(self, base_dir):
        return OutputManager(base_dir)
    
    def create_inference_engine(self, config):
        return InferenceEngine(config)
```

#### Step 3: Add BenchForge to Your Benchmark

```toml
# Your benchmark's pyproject.toml
[dependencies]
benchforge = {path = "../benchforge"}  # As submodule
# OR
benchforge = "^0.1.0"  # From PyPI
```

#### Step 4: Create Adapter in Your Benchmark

```python
# your_benchmark/adapter.py
from bench_forge.api import BenchForgeAPI

class BenchmarkAdapter:
    """Adapter to use BenchForge infrastructure"""
    
    def __init__(self, use_benchforge=True):
        self.use_benchforge = use_benchforge
        if use_benchforge:
            self.api = BenchForgeAPI()
        
    def get_llm_client(self, config):
        if self.use_benchforge:
            return self.api.create_llm_client(config)
        else:
            # Fallback to your legacy client
            return LegacyClient(config)
```

#### Step 5: Migrate Tasks

```python
# your_benchmark/tasks/task1.py
from your_benchmark.adapter import BenchmarkAdapter

class Task1:
    def __init__(self, use_benchforge=True):
        self.adapter = BenchmarkAdapter(use_benchforge)
        
    def run_inference(self, dataset, config):
        # Use BenchForge infrastructure
        client = self.adapter.get_llm_client(config)
        output_mgr = self.adapter.get_output_manager("results/task1")
        
        # Task-specific logic stays in your benchmark
        prompts = self.generate_prompts(dataset)
        responses = client.complete_batch(prompts)
        results = self.extract_results(responses)
        
        # Save using BenchForge output manager
        output_mgr.save_results(results)
        return results
```

## API Reference

### Core Components

#### LLM Client

```python
from bench_forge.llm.client import LLMClient
from bench_forge.llm.config import LLMConfig

config = LLMConfig(
    provider="openai",
    model="gpt-4",
    max_tokens=256,
    temperature=0.0
)

client = LLMClient(config)
response = client.complete("Your prompt here")
batch_responses = client.complete_batch(["prompt1", "prompt2"])
```

#### Output Manager

```python
from bench_forge.output.manager import OutputManager

manager = OutputManager(base_dir="results/")
manager.save_results(results_df, metadata={"model": "gpt-4"})
manager.save_metrics(metrics_df)
```

#### Inference Engine

```python
from bench_forge.engine import InferenceEngine

engine = InferenceEngine(config)
results = engine.run(
    dataset=dataset,
    prompt_fn=lambda x: f"Process: {x}",
    batch_size=20
)
```

## Best Practices

### For BenchForge Maintainers

1. **Keep it Generic**: Never add benchmark-specific code
2. **Clean API**: Maintain backward compatibility
3. **Documentation**: Update docs when adding features
4. **Testing**: Test infrastructure independently

### For Benchmark Developers

1. **Use the Adapter Pattern**: Create adapters for flexibility
2. **Keep Logic Separate**: Benchmark logic stays in your code
3. **Store Results Locally**: Results belong to your benchmark
4. **Version Compatibility**: Pin BenchForge version for stability

## Directory Structure

### Recommended BenchForge Structure

```
benchforge/
├── bench_forge/
│   ├── __init__.py
│   ├── api.py              # Clean public API
│   ├── engine/             # Inference engine
│   ├── llm/                # LLM abstraction
│   ├── output/             # Output management
│   ├── metrics/            # Metric calculation
│   ├── prompts/            # Prompt engineering
│   └── plugins/            # Plugin system
├── tests/
├── docs/
│   ├── ARCHITECTURE.md     # Technical architecture
│   ├── INTEGRATION_GUIDE.md # This file
│   └── API_REFERENCE.md    # Detailed API docs
└── pyproject.toml
```

### Recommended Benchmark Structure

```
your_benchmark/
├── src/your_benchmark/
│   ├── tasks/              # Task implementations
│   ├── infrastructure/     # Optional legacy code
│   └── benchforge_adapter.py # BenchForge adapter
├── results/                # Your results
├── configs/
└── pyproject.toml         # Includes benchforge
```

## Environment Configuration

```bash
# Enable/disable BenchForge in your benchmark
export USE_BENCHFORGE=true

# BenchForge plugin discovery
export BENCHFORGE_PLUGIN_PATH=/path/to/plugins
```

## Next Steps

- Review the [Architecture Documentation](./ARCHITECTURE.md)
- Check [API Reference](./API_REFERENCE.md) for detailed usage
- See [examples/](../examples/) for implementation patterns
- Read [FLAME Integration Guide](./FLAME_GUIDE.md) for FLAME-specific details