# Test Output Directory Fix Summary

## Problem
Test artifacts were being created in `evaluations/dummy/` instead of `tests/test_outputs/`.

## Root Causes

1. **Module-level imports**: evaluate.py and miscellaneous.py were importing EVALUATION_DIR and TEST_OUTPUT_DIR at module level, capturing values before conftest.py patches could be applied.

2. **Direct file saving**: Some evaluation modules (like casual_detection_evaluate_llm.py) were:
   - Creating directories with `evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)`
   - Saving files directly with `df.to_csv(evaluation_results_path, index=False)`

3. **Unnecessary directory creation**: Some evaluation modules (like fpb_evaluate.py) were creating directories even though they didn't save files.

## Fixes Applied

1. **Dynamic imports in evaluate.py**:
   - Changed from module-level imports to runtime imports inside functions
   - This ensures patched values from conftest.py are used

2. **Dynamic imports in miscellaneous.py**:
   - Moved config imports inside generate_inference_filename function
   - Ensures TEST_OUTPUT_DIR is resolved at runtime

3. **Removed directory creation from evaluation modules**:
   - fpb_evaluate.py: Removed path definition and mkdir call
   - casual_detection_evaluate_llm.py: Removed path definition, mkdir, and direct file saving

4. **Updated conftest.py**:
   - Removed directory patching since we want tests to use actual tests/test_outputs/
   - Only sets PYTEST_RUNNING=1 environment variable

## Result
All test artifacts are now correctly created in `tests/test_outputs/` only.