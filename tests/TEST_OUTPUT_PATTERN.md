# Test Output Pattern for FLaME

## Quick Summary

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