# FLaME Testing Guide

This document describes the testing infrastructure in FLaME (Financial Language Models Evaluation) project. It covers how tests are organized, how the test fixtures work, how to run tests, and recent changes to the testing framework.

## Test Organization

FLaME test files are located in the `/tests` directory:

- `conftest.py`: Global pytest fixtures and mocks
- `test_multi_inference.py`: Tests for multi-task inference
- `test_multi_evaluation.py`: Tests for multi-task evaluation
- `test_all_inference.py`: Parametrized tests that run all inference modules
- `test_all_evaluation.py`: Parametrized tests that run all evaluation modules
- `test_run_tasks_errors.py`: Tests for error handling in multi-task execution
- `test_task_validation.py`: Tests for task validation functionality
- `test_yaml_parsing.py`: Tests for YAML configuration parsing

## Test Fixtures and Mocking

Testing in FLaME follows these key principles:

1. Tests should run offline (no external API calls)
2. Tests should be fast
3. Test output should be isolated and not pollute production directories

The `conftest.py` file contains several important fixtures and mocks:

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

### Data Fixtures

- `dummy_args`: Provides a SimpleNamespace with default arguments
- `_DummyDataset`: Provides a simple in-memory dataset for testing

## Running Tests

Run all tests:
```
uv run python -m pytest
```

Run specific test file:
```
uv run python -m pytest tests/test_multi_inference.py
```

With verbosity:
```
uv run python -m pytest -v tests/test_multi_inference.py
```

## Recent Testing Changes (May 2025)

### Test Output Isolation Fix

The test execution previously wrote files to the main `/results` and `/evaluation` directories, polluting them with test data. The following changes were made to fix this:

1. **Updated `conftest.py`**:
   - Added `PYTEST_RUNNING=1` environment variable
   - Organized temporary directory creation and patching

2. **Updated `config.py`**:
   - Added `TEST_OUTPUT_DIR` as a dedicated test output directory
   - Added `IN_PYTEST` flag to detect test execution
   - Ensured all directories are created at startup

3. **Updated `inference.py`**:
   - Modified to check for test mode and use `TEST_OUTPUT_DIR` when in tests
   - Improved directory organization with task-specific subdirectories

4. **Updated `evaluate.py`**:
   - Similar changes to redirect output to test directories
   - Better path handling for results and metrics files

These changes ensure test outputs go to `/tests/test_outputs` (or a temp directory during pytest) instead of production directories.

## Test Files Pattern

Tests typically follow these patterns:

1. **Smoke Tests**: Verify code runs without errors, using mock responses (used in `test_all_*` files)
2. **Parametrized Tests**: Run the same test with multiple inputs (e.g., different task lists)
3. **Fixture-Based Testing**: Relying on the mock fixtures rather than real data/APIs

## Best Practices

1. **Always Use Fixtures**: Leverage the fixtures in `conftest.py` rather than creating new mocks
2. **Avoid External Calls**: Tests should never make real API calls
3. **Check Test Output**: Verify outputs go to test directories, not production folders
4. **Use test_output_dir**: For any new code that saves files, make sure it respects `IN_PYTEST` and uses the test directory

## Adding New Tests

When adding new tests:

1. Create appropriate test file in `/tests` directory
2. Use the existing fixtures from `conftest.py`
3. Make sure outputs use the test directories
4. Run tests in isolation before submitting changes