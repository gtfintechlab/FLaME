# FLaME Project Structure

This document describes the organization and structure of the FLaME (Financial Language Understanding Evaluation) framework.

## Root Directory

```
FLaME/
├── main.py                    # Main entry point for running tasks
├── setup.py                   # Package setup configuration
├── pyproject.toml            # Modern Python project configuration
├── requirements.txt          # Python dependencies
├── uv.lock                   # UV package manager lock file
├── pytest.ini                # Pytest configuration
├── .gitignore               # Git ignore patterns
├── LICENSE.md               # Project license
├── README.md                # Project overview and usage
├── CLAUDE.md                # Claude AI assistant instructions
└── task_validation_tracker.md # Task validation progress tracking
```

## Core Source Code (`/src/flame/`)

```
src/flame/
├── __init__.py
├── config.py                 # Global configuration settings
├── task_registry.py         # Maps task names to functions
├── code/                    # Task implementations
│   ├── inference.py         # Main inference orchestrator
│   ├── evaluate.py          # Main evaluation orchestrator
│   ├── extraction_prompts.py # Extraction prompts (legacy)
│   ├── tokens.py            # Token utilities
│   ├── prompts/             # Prompt management system
│   │   ├── __init__.py
│   │   ├── base.py          # Base prompt classes
│   │   ├── constants.py     # Prompt constants
│   │   ├── fewshot.py       # Few-shot prompts
│   │   ├── registry.py      # Prompt registry
│   │   └── zeroshot.py      # Zero-shot prompts
│   ├── <task_name>/         # Task-specific implementations
│   │   ├── __init__.py
│   │   ├── <task>_inference.py
│   │   └── <task>_evaluate.py
│   └── _archive/            # Archived/old implementations
└── utils/                   # Utility modules
    ├── __init__.py
    ├── batch_utils.py       # Batch processing utilities
    ├── dataset_utils.py     # Dataset loading helpers
    ├── logging_utils.py     # Logging configuration
    ├── miscellaneous.py     # Misc utilities
    └── output_utils.py      # Output file handling
```

## Configuration (`/configs/`)

Contains YAML configuration files for each task and special configurations:
- `default.yaml` - Default configuration
- `development.yaml` - Development config using Ollama
- `ollama.yaml` - Ollama-specific settings
- `<task_name>.yaml` - Task-specific configurations

## Data (`/data/`)

Raw datasets organized by task:
```
data/
├── <TaskName>/              # Task-specific data
│   ├── train.json/csv       # Training data
│   ├── test.json/csv        # Test data
│   ├── dev.json/csv         # Development/validation data
│   └── huggify_*.py         # Scripts to upload to HuggingFace
```

## Results & Evaluations

```
results/                     # Inference outputs
└── <task_name>/
    └── <provider>/
        └── <model>/
            └── <model>__<task>__r<run>__<date>__<hash>.csv

evaluations/                 # Evaluation outputs
└── <task_name>/
    └── <provider>/
        └── <model>/
            ├── <model>__<task>__r<run>__<date>__<hash>.csv
            └── <model>__<task>__r<run>__<date>__<hash>_metrics.csv
```

## Tests (`/tests/`)

```
tests/
├── conftest.py              # Test fixtures and configuration
├── fixtures/                # Test fixtures
│   └── ollama_fixtures.py   # Ollama-specific fixtures
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── modules/                 # Module-specific tests
│   ├── test_all_inference.py
│   └── test_all_evaluation.py
├── multi_task/              # Multi-task tests
└── prompts/                 # Prompt system tests
```

## Scripts (`/scripts/`)

Development and utility scripts:
```
scripts/
├── ollama/                  # Ollama testing scripts
│   ├── test_ollama_connection.py
│   └── test_phase2_with_ollama.py
└── validation/              # Validation utilities
    ├── check_phase2_status.py
    └── test_convfinqa_small.py
```

## Documentation (`/docs/`)

```
docs/
├── multi_task_guide.md      # Multi-task usage guide
├── prompt_examples.md       # Prompt examples
├── prompt_system.md         # Prompt system documentation
├── project_structure.md     # This file
├── OLLAMA.md               # Ollama integration guide
└── phase2/                  # Phase 2 validation docs
    └── phase2_validation_summary.md
```

## Claude Documentation (`/claude/`)

Contains testing guides and documentation for development:
- LiteLLM integration guides
- Optimization plans
- Testing documentation

## Logs (`/logs/`)

Runtime logs organized by component (excluded from git):
- `<component>_<task>.log` - Component-specific logs
- `flame.log` - Main application log

## Key Conventions

1. **Task Naming**: Use lowercase with underscores (e.g., `causal_detection`)
2. **File Naming**: `<task_name>_inference.py` and `<task_name>_evaluate.py`
3. **Result Files**: Include model, task, run number, date, and hash
4. **Logging**: Use component-based loggers for better organization
5. **Testing**: All tasks should have corresponding tests

## Adding a New Task

1. Create directory: `src/flame/code/<new_task>/`
2. Implement: `<new_task>_inference.py` and `<new_task>_evaluate.py`
3. Add prompts to `src/flame/code/prompts/`
4. Register in `src/flame/task_registry.py`
5. Create config: `configs/<new_task>.yaml`
6. Add tests to `tests/`
7. Document in `task_validation_tracker.md`