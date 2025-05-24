# Multi-Task Execution Guide for FLaME

Updated: 2025-05-14

This guide covers the multi-task functionality in FLaME, which allows running multiple tasks (inference or evaluation) with a single command. This is especially useful for benchmarking models across multiple financial language tasks or running comprehensive evaluations.

## 1. Introduction

### 1.1 What is Multi-Task Execution?

Multi-task execution enables you to run multiple FLaME tasks sequentially with a single command rather than executing each task separately. For example, you can:

- Run inference on multiple financial language tasks (FOMC, NumClaim, FinEntity, etc.)
- Evaluate model performance across multiple datasets
- Mix and match tasks for customized benchmarking

### 1.2 Architecture Overview

The multi-task functionality is implemented with these key components:

1. **Task Registry**: A central registry (`task_registry.py`) that maps task names to their implementation functions
2. **Configuration Management**: Support for both CLI arguments and YAML config files
3. **Sequential Executor**: A runner that executes tasks one by one and collects any errors
4. **Error Handling**: A `MultiTaskError` mechanism that aggregates errors for better debugging

## 2. Configuration

There are two ways to configure multi-task execution: via YAML config files or command-line arguments.

### 2.1 YAML Configuration

YAML configuration is recommended for complex setups or when you want to save your configuration:

```yaml
# configs/my_multitask.yaml
mode: inference  # or evaluate
tasks:  # List of tasks to execute
  - fomc
  - numclaim
  - finer
# Model parameters
model: "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
max_tokens: 128
temperature: 0.0
# Other parameters as needed
```

Run the configuration with:

```bash
python main.py --config configs/my_multitask.yaml
```

### 2.2 Command-Line Interface (CLI)

For quick execution, you can use command-line arguments:

```bash
python main.py --mode inference --tasks fomc numclaim finer --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
```

### 2.3 Configuration Precedence

When both YAML and CLI arguments are provided, the precedence order is:
1. Command-line arguments
2. YAML configuration values
3. Default values

This allows you to specify a base configuration in YAML and override specific parameters via the CLI.

### 2.4 Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mode` | Processing mode: `inference` or `evaluate` | None (required) |
| `tasks` | List of tasks to execute | None (required) |
| `file_name` | File name for evaluation (required for `evaluate` mode) | None |
| `model` | Model to use for inference | None |
| `max_tokens` | Maximum number of tokens to generate | 128 |
| `temperature` | Sampling temperature | 0.0 |
| `top_p` | Top-p sampling parameter | 0.9 |
| `top_k` | Top-k sampling parameter | None |
| `repetition_penalty` | Penalty for repetitive text | 1.0 |
| `batch_size` | Batch size for inference | 10 |
| `prompt_format` | Format of prompts (`zero_shot` or `few_shot`) | `zero_shot` |

## 3. Using Multi-Task for Inference

### 3.1 Basic Inference Example

To run inference on multiple tasks:

```bash
python main.py --mode inference --tasks numclaim fomc finentity --model "your_model_name"
```

### 3.2 Customized Inference Example

For more complex inference with custom parameters:

```yaml
# configs/custom_inference.yaml
mode: inference
tasks:
  - causal_classification
  - subjectiveqa
  - ectsum
model: "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
max_tokens: 256
temperature: 0.1
batch_size: 5
prompt_format: "few_shot"
```

Run with:

```bash
python main.py --config configs/custom_inference.yaml
```

### 3.3 Available Inference Tasks

The following tasks are currently supported for inference:

- `numclaim` - Numerical claim classification
- `fomc` - Federal Open Market Committee sentiment analysis
- `finer` - Financial named entity recognition
- `finentity` - Financial entity extraction
- `causal_classification` - Causal relationship classification
- `subjectiveqa` - Subjective question answering
- `ectsum` - Earnings call transcript summarization
- `fnxl` - Financial document extraction

## 4. Using Multi-Task for Evaluation

### 4.1 Basic Evaluation Example

To evaluate the results of multiple tasks:

```bash
python main.py --mode evaluate --tasks numclaim finer --file_name results.csv
```

### 4.2 Customized Evaluation Example

For more complex evaluation:

```yaml
# configs/custom_evaluation.yaml
mode: evaluate
tasks:
  - numclaim
  - finentity
  - fnxl
file_name: model_outputs_2025_05_10.csv
batch_size: 20
```

Run with:

```bash
python main.py --config configs/custom_evaluation.yaml
```

### 4.3 Available Evaluation Tasks

The following tasks are currently supported for evaluation:

- `numclaim` - Numerical claim classification
- `finer` - Financial named entity recognition
- `finentity` - Financial entity extraction
- `fnxl` - Financial document extraction
- `causal_classification` - Causal relationship classification
- `subjectiveqa` - Subjective question answering
- `ectsum` - Earnings call transcript summarization
- `refind` - Financial document relation extraction
- `banking77` - Banking intent classification
- `convfinqa` - Conversational financial QA
- `finqa` - Financial question answering
- `tatqa` - Table and text question answering
- `causal_detection` - Causal relation detection

## 5. Error Handling

The multi-task system uses a robust error handling approach:

1. Tasks are executed sequentially
2. If a task fails, execution continues with the next task
3. All errors are collected in a `MultiTaskError` exception
4. At the end of execution, if any tasks failed, the `MultiTaskError` is raised with details about which tasks failed and why

### 5.1 Error Reporting Example

When errors occur, you'll see output like:

```
MultiTaskError: Errors in tasks: fomc, finentity

Task: fomc
Error: ValueError: Invalid sentiment label detected in results

Task: finentity
Error: FileNotFoundError: Could not find entity dictionary at /path/to/dict.json
```

This approach ensures that:
- You can run multiple tasks without stopping at the first failure
- You get a comprehensive error report showing all issues at once
- The system maintains appropriate error propagation for CI/CD pipelines

## 6. Best Practices

### 6.1 Task Selection

- **Group related tasks**: Run semantically related tasks together (e.g., all entity extraction tasks)
- **Balance task complexity**: Mix lightweight and heavyweight tasks for better resource utilization
- **Use task validation**: The system validates task names before execution; check supported tasks with `from flame.task_registry import supported`

### 6.2 Configuration Management

- **Create task-specific configs**: Maintain separate YAML files for different task groups
- **Parameterize common values**: Use environment variables or shared config files for common parameters
- **Version your configs**: Include configuration files in version control to track changes

### 6.3 Resource Management

- **Be mindful of batch sizes**: Larger batch sizes use more memory but may be faster
- **Control concurrency**: For now, tasks run sequentially; plan accordingly for long-running tasks
- **Monitor resource usage**: Watch memory consumption when running many tasks together

### 6.4 Error Handling

- **Check task validity first**: Validate task names before running lengthy processes
- **Handle partial failures**: Design workflows to handle cases where some tasks succeeded and others failed
- **Understand error context**: The `MultiTaskError` provides a dictionary of task-specific exceptions

### 6.5 CI/CD Integration

- **Use exit codes**: The script will exit with a non-zero code if any task fails
- **Script automation**: Create wrapper scripts that parse the error output for automated reporting
- **Run smoke tests**: Use a subset of tasks for quick verification before full runs

## 7. Troubleshooting

### 7.1 Common Issues

| Problem | Possible Solution |
|---------|------------------|
| `Task 'xyz' not supported` | Check spelling or use `from flame.task_registry import supported` to list available tasks |
| `File name is required for evaluation mode` | Add `--file_name your_results.csv` for evaluation |
| Memory errors | Reduce batch size or split tasks into smaller groups |
| API rate limiting | Add delays between tasks or reduce batch size |

### 7.2 Debugging Tips

- Run single tasks first to isolate issues
- Use the `--config` option to save and reproduce problematic configurations 
- Check task-specific logs in the `logs/` directory
- Examine generated outputs in the `results/` directory

## 8. Extending the System

To add a new task to the multi-task system:

1. Implement your task's inference and/or evaluation functions
2. Register them in `task_registry.py` by adding to `INFERENCE_MAP` and/or `EVALUATE_MAP`
3. Update tests as needed

Example registration:

```python
# In task_registry.py
from flame.code.my_task.my_task_inference import my_task_inference
from flame.code.my_task.my_task_evaluate import my_task_evaluate

INFERENCE_MAP: dict[str, callable] = {
    # ... existing tasks
    "my_task": my_task_inference,
}

EVALUATE_MAP: dict[str, callable] = {
    # ... existing tasks
    "my_task": my_task_evaluate,
}
```

---

## Appendix: Task Reference

| Task Name | Description | Inference | Evaluation |
|-----------|-------------|:---------:|:----------:|
| `numclaim` | Numerical claim classification | ✓ | ✓ |
| `fomc` | Fed Open Market Committee sentiment | ✓ | - |
| `finer` | Financial named entity recognition | ✓ | ✓ |
| `finentity` | Financial entity extraction | ✓ | ✓ |
| `causal_classification` | Causal relationship classification | ✓ | ✓ |
| `subjectiveqa` | Subjective financial QA | ✓ | ✓ |
| `ectsum` | Earnings call transcript summarization | ✓ | ✓ |
| `fnxl` | Financial document extraction | ✓ | ✓ |
| `refind` | Financial document relation extraction | - | ✓ |
| `banking77` | Banking intent classification | - | ✓ |
| `convfinqa` | Conversational financial QA | - | ✓ |
| `finqa` | Financial question answering | - | ✓ |
| `tatqa` | Table and text question answering | - | ✓ |
| `causal_detection` | Causal relation detection | - | ✓ |