# FLaME Testing Suite

## Overview

This directory contains all tests for the FLaME (Financial Language Understanding Evaluation) framework. The testing suite is designed to run quickly and offline, with all external dependencies mocked.

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

### Directory Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Global test configuration
├── test_outputs/                # All test artifacts (gitignored)
│   ├── results/                 # Inference results
│   ├── evaluation/              # Evaluation results
│   └── logs/                    # Test logs
├── test_all_inference.py        # Test all inference modules
├── test_all_evaluation.py       # Test all evaluation modules
├── test_multi_task_integration.py # Multi-task integration tests
└── ...                          # Other test modules
```

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

## Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_all_inference.py

# Run with verbose output
uv run pytest -vv

# Run specific test
uv run pytest -k "test_name"
```

## Debugging

If you need to inspect test outputs:
1. Run the test
2. Check `tests/test_outputs/` for generated files
3. Files are organized by module/task name

## Notes

- All test outputs are temporary and not committed to version control
- The testing framework ensures isolation between tests
- External API calls are mocked for speed and reliability

## Quick Summary of Test Output Pattern

**All test artifacts MUST go to `tests/test_outputs/`** which is gitignored.

## How It Works

1. **conftest.py** sets `PYTEST_RUNNING=1` environment variable
2. **flame.config** module defines:
   - `IN_PYTEST` = checks for `PYTEST_RUNNING` env var
   - `TEST_OUTPUT_DIR` = `tests/test_outputs/`
3. **inference.py** and **evaluate.py** use:
   ```python
   output_dir = TEST_OUTPUT_DIR if IN_PYTEST else RESULTS_DIR
   ```

## For AI Agents and Developers

### ✅ DO:
- Use the existing pattern in inference.py and evaluate.py
- Store all test artifacts in `TEST_OUTPUT_DIR`
- Let conftest.py handle the redirection automatically

### ❌ DON'T:
- Hardcode paths to `results/` or `evaluations/` in tests
- Create output files outside of `TEST_OUTPUT_DIR`
- Commit any files from `tests/test_outputs/`

### Pattern Example:

```python
from flame.config import IN_PYTEST, TEST_OUTPUT_DIR, RESULTS_DIR

# This pattern is ALREADY IMPLEMENTED in inference.py and evaluate.py
output_dir = TEST_OUTPUT_DIR if IN_PYTEST else RESULTS_DIR
task_dir = output_dir / task_name
task_dir.mkdir(parents=True, exist_ok=True)
```

## Directory Structure

```
tests/
├── test_outputs/           # ALL test artifacts (gitignored)
│   ├── results/           # Inference outputs
│   ├── evaluation/        # Evaluation outputs
│   └── logs/              # Test logs
├── conftest.py            # Sets up test environment
├── README.md              # Detailed testing guide
└── TEST_OUTPUT_PATTERN.md # This file
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