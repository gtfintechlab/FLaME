# FLaME Test Suite

This directory contains all tests for the FLaME (Financial Language Understanding Evaluation) framework. The testing suite is designed to run quickly and offline, with all external dependencies mocked.

## Directory Structure

```
tests/
├── conftest.py                # Global pytest configuration and fixtures
├── test_outputs/              # All test artifacts (gitignored)
│   ├── results/               # Inference outputs
│   ├── evaluation/            # Evaluation outputs
│   └── logs/                  # Test logs
├── unit/                      # Unit tests for core functionality
│   ├── test_task_validation.py    # Task registry validation
│   ├── test_yaml_parsing.py       # YAML configuration parsing
│   ├── test_output_directory.py   # Output directory configuration
│   └── test_run_tasks_errors.py   # Error handling (MultiTaskError)
│
├── modules/                   # Tests for all task modules
│   ├── test_all_inference.py      # Parametrized tests for all inference modules
│   └── test_all_evaluation.py     # Parametrized tests for all evaluation modules
│
├── prompts/                   # Prompt system tests
│   ├── test_prompt_registry.py    # Prompt registry functionality
│   ├── test_prompt_aliases.py     # Prompt alias handling
│   └── test_prompts_package.py    # Prompts package imports and structure
│
├── multi_task/                # Multi-task execution tests
│   ├── test_multi_inference.py    # Multi-task inference functionality
│   └── test_multi_evaluation.py   # Multi-task evaluation functionality
│
└── integration/               # Integration and end-to-end tests
    └── test_multi_task_integration.py  # Complete multi-task workflow tests
```

## Test Categories

### Unit Tests (`unit/`)
Core functionality tests that verify individual components work correctly in isolation:
- Task validation and registry
- YAML parsing and configuration
- Output directory management
- Error handling mechanisms

### Module Tests (`modules/`)
Comprehensive smoke tests that verify all inference and evaluation modules can be imported and executed:
- Uses parametrized testing to discover and test all modules
- All external dependencies are mocked (no real API calls)
- Ensures basic functionality of every task module

### Prompt Tests (`prompts/`)
Tests for the prompt system:
- Registry lookups and function mapping
- Direct imports and aliases
- Prompt function behavior

### Multi-Task Tests (`multi_task/`)
Tests specifically for multi-task execution features:
- Running multiple inference tasks
- Running multiple evaluation tasks
- Task sequencing and error handling

### Integration Tests (`integration/`)
End-to-end workflow tests that verify complete user scenarios:
- YAML configuration to execution
- Command-line argument handling
- Error recovery and reporting
- Progress tracking

## Architecture

### Key Components

1. **conftest.py**: Global pytest configuration that:
   - Mocks external dependencies (litellm, datasets, etc.)
   - Redirects all output directories to temporary folders
   - Sets `PYTEST_RUNNING` environment variable
   - Ensures tests run offline and fast

2. **Test Output Handling**:
   - When `IN_PYTEST` is True, all outputs go to `TEST_OUTPUT_DIR`
   - Normal operation uses `RESULTS_DIR` and `EVALUATION_DIR`
   - This is handled automatically by `flame.config`

### Key Mocks

- `litellm.completion/batch_completion`: Replaced with static mock responses
- `datasets.load_dataset`: Replaced with small in-memory dataset
- `time.sleep`: No-op to speed up tests
- External APIs (Together AI client, etc): Mocked to avoid network calls

### Output Redirection

Tests redirect all output to isolated test directories:
- RESULTS_DIR → tmp_path/results
- LOG_DIR → tmp_path/logs
- EVALUATION_DIR → tmp_path/evaluation
- TEST_OUTPUT_DIR → tmp_path/test_outputs

This is implemented through:
1. Setting `PYTEST_RUNNING=1` environment variable
2. The `IN_PYTEST` flag in `config.py`
3. Conditional output paths in both `inference.py` and `evaluate.py`

## Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test category
uv run pytest tests/unit/
uv run pytest tests/modules/
uv run pytest tests/integration/

# Run specific test file
uv run pytest tests/unit/test_task_validation.py

# Run with verbose output
uv run pytest -vv

# Run tests matching a pattern
uv run pytest -k "multi_task"

# Run with interactive debugging on failure
uv run pytest --pdb
```

## Test Output Directory

All test artifacts (CSV files, logs, evaluation results, etc.) are automatically stored in:
```
tests/test_outputs/
```

This directory is:
- Created automatically when tests run
- Ignored by git (see `.gitignore`)
- Used by all test modules when `IN_PYTEST` is detected
- Cleaned up automatically after each test session

## Best Practices

### For AI Agents and Developers

1. **Never commit test outputs**: The `test_outputs/` directory is gitignored
2. **Use existing patterns**: The output redirection is automatic when running tests
3. **Check IN_PYTEST**: Use `flame.config.IN_PYTEST` to detect test environment
4. **Output directories**: 
   - In tests: Use `TEST_OUTPUT_DIR`
   - In production: Use `RESULTS_DIR` or `EVALUATION_DIR`

### Adding New Tests

When creating new tests that generate files:

```python
from flame.config import IN_PYTEST, TEST_OUTPUT_DIR, RESULTS_DIR

# This pattern is already used in inference.py and evaluate.py
output_dir = TEST_OUTPUT_DIR if IN_PYTEST else RESULTS_DIR
```

### Example Test Pattern

```python
def test_my_feature(dummy_args):
    """Test that generates output files"""
    # The conftest.py automatically redirects outputs
    # Just use the normal inference/evaluation functions
    
    # For custom file outputs, follow this pattern:
    from flame.config import TEST_OUTPUT_DIR
    
    output_file = TEST_OUTPUT_DIR / "my_test" / "output.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write your test output
    df.to_csv(output_file, index=False)
```

### Test Patterns

Tests typically follow these patterns:

1. **Smoke Tests**: Verify code runs without errors, using mock responses (used in `test_all_*` files)
2. **Parametrized Tests**: Run the same test with multiple inputs (e.g., different task lists)
3. **Fixture-Based Testing**: Relying on the mock fixtures rather than real data/APIs

## Debugging

If you need to inspect test outputs:
1. Run the test
2. Check `tests/test_outputs/` for generated files
3. Files are organized by module/task name

## Important Notes

1. **No Live API Calls**: All tests use mocked external dependencies (defined in `conftest.py`)
2. **Test Outputs**: All test artifacts are automatically redirected to `tests/test_outputs/` (gitignored)
3. **Fast Execution**: Tests are designed to run quickly with minimal dependencies
4. **Automatic Discovery**: Module tests automatically discover new inference/evaluation modules
5. **External API calls are mocked**: Tests should never make real API calls
6. **Test isolation**: The testing framework ensures isolation between tests

## How It Works

1. **conftest.py** sets `PYTEST_RUNNING=1` environment variable
2. **flame.config** module defines:
   - `IN_PYTEST` = checks for `PYTEST_RUNNING` env var
   - `TEST_OUTPUT_DIR` = `tests/test_outputs/`
3. **inference.py** and **evaluate.py** use:
   ```python
   output_dir = TEST_OUTPUT_DIR if IN_PYTEST else RESULTS_DIR
   ```

## Key Files

1. **src/flame/config.py**: Defines `IN_PYTEST` and `TEST_OUTPUT_DIR`
2. **src/flame/code/inference.py**: Uses pattern at line 40
3. **src/flame/code/evaluate.py**: Uses pattern at line 38
4. **tests/conftest.py**: Sets up test environment

## Why This Pattern?

- Keeps repository clean
- Prevents accidental commits of test data
- Ensures consistent test behavior
- Makes tests portable and reproducible