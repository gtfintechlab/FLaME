# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Additional Instructions
- @~/.claude/epic1/epic1_plan.md
- @~/.claude/epic1/multi_task_guide.md

## Project Overview

FLaME (Financial Language Understanding Evaluation) is a framework for evaluating Large Language Models (LLMs) on financial tasks. The repository contains multiple datasets for evaluating model performance on tasks like sentiment analysis, causal classification, summarization, and more in the financial domain.

## Key Commands

### Project Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# Or
.\.venv\Scripts\activate  # On Windows

# Install dependencies using uv (preferred)
uv pip install -r requirements.txt

# Install FLaME in development mode
uv pip install -e .

# Test the installation
uv run python -c "import flame; print(flame.__file__)"
```

### Running Inference

Run inference on a dataset using a specific model:

```bash
# Using uv
uv run python main.py --config configs/default.yaml --mode inference --dataset <dataset_name> --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
```

Example:
```bash
uv run python main.py --config configs/default.yaml --mode inference --dataset fomc --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
```

Override parameters from config file:
```bash
uv run python main.py --config configs/default.yaml --mode inference --dataset fomc --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct" --max_tokens 256 --temperature 0.1 --prompt_format zero_shot
```

### Running Evaluation

Run evaluation on inference results:

```bash
uv run python main.py --config configs/default.yaml --mode evaluate --dataset <dataset_name> --file_name <results_file_path>
```

Example:
```bash
uv run python main.py --config configs/default.yaml --mode evaluate --dataset fomc --file_name "results/fomc/fomc_together_ai_meta-llama_Llama-4-Scout-17B-16E-Instruct_13_05_2025.csv"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_all_inference.py

# Run specific test with verbose output
uv run pytest -vv tests/test_all_inference.py

# Run specific test matching a pattern
uv run pytest -k "fomc"

# Run test in debug mode
uv run pytest --pdb
```

### Installing Packages with uv

Always use `uv` for package management:

```bash
# Add a new package
uv add packagename

# Add a package with specific version
uv add packagename==1.2.3

# Run Python scripts
uv run python script.py

# Generate requirements.txt
uv pip compile pyproject.toml -o requirements.txt
```

## Project Architecture

### Core Components

1. **Data Management**: 
   - The `data/` directory contains subfolders for each dataset
   - Each dataset folder contains a `huggify_*.py` script to upload data to HuggingFace

2. **Inference Pipeline**:
   - `main.py`: Entry point that parses args and runs inference or evaluation
   - `src/flame/code/inference.py`: Core orchestrator for inference tasks
   - `src/flame/code/evaluate.py`: Core orchestrator for evaluation tasks
   - `src/flame/code/<dataset>/<dataset>_inference.py`: Task-specific inference logic
   - `src/flame/code/<dataset>/<dataset>_evaluate.py`: Task-specific evaluation logic
   - `src/flame/code/prompts.py`: Zero-shot prompts used for inference

3. **Configuration System**:
   - `configs/default.yaml`: Default configuration settings
   - Command-line arguments can override config file values
   - Configuration flows through application via args object

4. **Testing Framework**:
   - Smoke tests for all inference and evaluation modules
   - Mock/stub system in `conftest.py` ensures tests run offline and fast

### Directory Structure

```
FLaME/
├── configs/                # Configuration files
├── data/                   # Task-specific datasets
├── src/flame/              # Main source code
│   ├── code/               # Task implementations
│   │   ├── <dataset>/      # Dataset-specific code
│   │   │   ├── <dataset>_inference.py
│   │   │   └── <dataset>_evaluate.py
│   │   ├── inference.py    # Main inference orchestrator
│   │   ├── evaluate.py     # Main evaluation orchestrator
│   │   └── prompts.py      # Prompt templates
│   ├── utils/              # Utility modules
│   └── config.py           # Project-wide configuration
├── results/                # Inference results 
├── evaluations/            # Evaluation results
├── tests/                  # Test suite
│   ├── conftest.py         # Test fixtures and mocks
│   ├── test_all_inference.py # Test all inference modules
│   └── test_all_evaluation.py # Test all evaluation modules
└── claude/                 # Testing guides and documentation
```

### Code Flow

1. **Inference pipeline**:
   - `main.py` parses args and loads config
   - `inference.py` maps dataset name to task-specific inference function
   - Task function loads dataset, creates prompts, calls the LLM API, and returns a DataFrame
   - Results are saved to CSV in `results/<dataset>/`

2. **Evaluation pipeline**:
   - `main.py` parses args and loads config
   - `evaluate.py` maps dataset name to task-specific evaluation function
   - Evaluation function loads inference results, computes metrics, and returns DataFrames
   - Results are saved to CSV in `evaluations/<dataset>/`

## API Integration

The project uses LiteLLM as an abstraction layer for calling different LLM providers. By default, use the Together.ai provider with the Llama-4-Scout-17B-16E-Instruct model:

```yaml
model: "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
```

## Testing Framework

FLaME has a comprehensive testing framework:

1. **Test Discovery**: 
   - Tests dynamically discover inference and evaluation modules

2. **Stubbed External Calls**: 
   - LiteLLM API calls are mocked/stubbed
   - No network connections or API keys needed for tests

3. **Test Isolation**:
   - Tests run in isolation with temporary directories
   - Fast and deterministic

4. **Debug Helpers**:
   - Verbose test outputs with `-vv`
   - Interactive debugging with `--pdb`

## Testing Guides

For detailed guidance on the testing framework, refer to these resources:

- [Inference Testing Guide](./claude/inference_testing_guide.md): Details on testing inference modules
- [Evaluation Testing Guide](./claude/evaluation_testing_guide.md): Details on testing evaluation modules
- [Testing Suite Notes](./claude/testing_suite_notes.md): Design and roadmap for testing

## Common Patterns

1. **Adding a new dataset**:
   - Create a new folder in `data/<new_dataset>/`
   - Create a `huggify_<new_dataset>.py` script if uploading to HuggingFace
   - Add task-specific code in `src/flame/code/<new_dataset>/`
   - Implement `<new_dataset>_inference.py` and `<new_dataset>_evaluate.py`
   - Add prompt generation in `src/flame/code/prompts.py`
   - Update the task map in `src/flame/code/inference.py` and `src/flame/code/evaluate.py`

2. **Creating prompts**:
   - Define a function in `prompts.py` that takes input text and returns formatted prompt
   - Follow existing pattern like:
     ```python
     def task_prompt(input_text: str) -> str:
         return f"""Instructions for task.
                    Input: {input_text}"""
     ```

3. **Working with results**:
   - Results are saved as CSVs in `results/<dataset>/`
   - Follow naming convention: `<dataset>_<provider>_<model>_<date>.csv`
   - Evaluation metrics are saved in `evaluations/<dataset>/`

4. **Batch processing**:
   - Use `chunk_list` from `flame.utils.batch_utils` to create batches
   - Process with `process_batch_with_retry` for resilient API calls

## Important Notes

1. **API Keys**:
   - Store API keys in a `.env` file in the project root
   - Required keys include:
     ```
     HUGGINGFACEHUB_API_TOKEN=<your_token>
     TOGETHER_API_KEY=<your_key>
     ```

2. **Default Model**:
   - Always use the Together.ai provider with Llama-4-Scout-17B-16E-Instruct model
   ```
   together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct
   ```

3. **Package Management**:
   - Always use `uv` for package management and running Python scripts
   - Examples:
     ```bash
     # Install packages
     uv add packagename
     
     # Run Python scripts
     uv run python script.py
     ```

4. **Testing**:
   - Run tests before making changes to ensure everything works
   - Tests use mock/stub libraries to avoid network calls

5. **Result Paths**:
   - Inference results: `results/<dataset>/<dataset>_<provider>_<model>_<date>.csv`
   - Evaluation results: `evaluations/<dataset>/evaluation_<dataset>_<model>_<date>.csv`

## Project Specific Memories
- do not use `uv run python main.py` ever on your own, let me run that command