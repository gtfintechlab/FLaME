# FLaME Evaluation Module Issues

This document captures evaluation module issues discovered during testing on May 14, 2025. These issues should be addressed in a future sprint to ensure the full test suite passes.

## 1. Summary of Failed Evaluation Tests

During our testing, 5 out of 72 tests failed, specifically in the following evaluation modules:

1. `flame.code.banking77.banking77_evaluate`
2. `flame.code.convfinqa.convfinqa_evaluate`
3. `flame.code.ectsum.ectsum_evaluate`
4. `flame.code.edtsum.edtsum_evaluate`
5. `flame.code.tatqa.tatqa_evaluate`

## 2. Detailed Error Analysis

### 2.1. `banking77_evaluate`

The specific error message was not fully displayed in the test output, but the test failed during execution. This module likely has issues with how it processes the dummy data provided in the test fixtures.

**Priority:** Medium - Banking77 is not one of the core financial tasks.

### 2.2. `convfinqa_evaluate`

Error message:
```
2025-05-14 14:42:08,176 - convfinqa_evaluation - INFO - Starting evaluation for dummy using model gpt-3.5-turbo...
2025-05-14 14:42:08,176 - convfinqa_evaluation - INFO - Loaded data from dummy.csv for evaluation.
2025-05-14 14:42:08,176 - convfinqa_evaluation - ERROR - Error processing response 0: This function is deprecated. Do not pass stop tokens with LiteLLM.
2025-05-14 14:42:08,177 - convfinqa_evaluation - INFO - CSV updated at iteration 1/1
```

**Issue:** The evaluator is using a deprecated function with LiteLLM, specifically passing stop tokens in a way that's no longer supported.

**Priority:** High - ConvFinQA is an important financial task and this reflects an API compatibility issue.

### 2.3. `ectsum_evaluate`

The specific error message was not fully displayed in the test output, but the test failed during execution. This is likely related to how the ECTSUM evaluator processes dummy data or calculates metrics.

**Priority:** High - ECTSUM is one of the core financial summarization tasks.

### 2.4. `edtsum_evaluate`

The specific error message was not fully displayed in the test output, but the test failed during execution. Similar to ECTSUM, this is likely related to data processing or metric calculation.

**Priority:** Medium - Another summarization task but with less priority than ECTSUM.

### 2.5. `tatqa_evaluate`

Error message:
```
Failed: flame.code.tatqa.tatqa_evaluate crashed: Classification metrics can't handle a mix of binary and unknown targets
```

**Issue:** The evaluator is trying to use scikit-learn's classification metrics with incompatible target formats. It's likely passing a mix of binary and unknown/NaN values to a metric function that can't handle this mixture.

**Priority:** High - TATQA is an important question-answering task for financial tables.

## 3. Runtime Warnings

The following warnings were also observed during testing:

```
RuntimeWarning: Mean of empty slice.
  avg = a.mean(axis, **keepdims_kw)

RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
```

These warnings appeared in tests for:
- `flame.code.causal_classification.causal_classification_evaluate`
- `flame.code.causal_detection.casual_detection_evaluate_llm`

**Issue:** These modules are attempting to calculate means of empty arrays or performing divisions that result in invalid values (likely dividing by zero).

**Priority:** Low - These are warnings, not errors, and the tests still pass.

## 4. Recommended Fixes

### 4.1. General Fixes

1. **Enhance Test Fixtures:**
   - Update `_DummyRow` and `_DummyDataset` classes in `conftest.py` to provide more realistic data for each evaluation task.
   - Add task-specific dummy data generators for complex formats like TATQA's tabular data.

2. **Improve Error Handling:**
   - Add defensive checks for empty datasets and edge cases.
   - Ensure all metric calculations have proper error handling.

### 4.2. Task-Specific Fixes

#### Banking77
- Inspect how dummy data is processed in the evaluator.
- Check for any dependencies on specific fields in the test data.

#### ConvFinQA
- Update the LiteLLM API calls to no longer pass stop tokens.
- Review the most recent LiteLLM documentation for proper API usage.

#### ECTSUM/EDTSUM
- Add proper dummy summaries to the test data.
- Ensure the ROUGE or BERTScore metric calculations can handle empty or dummy text.

#### TATQA
- Fix the classification metric handler to properly handle unknown targets.
- Use a try-except block around metric calculations to prevent crashes.
- Consider using `sklearn.metrics.classification_report` with `zero_division=0` parameter.

## 5. Future Test Improvements

1. **Enhanced Isolation:**
   - Further isolate each evaluation module by creating specialized mock objects for external dependencies.
   - Mock complex libraries like BERT-score more completely.

2. **Better Test Documentation:**
   - Add more detailed docstrings to test functions explaining what is being tested.
   - Document the expected format of dummy data for each evaluation module.

3. **Fixture Improvements:**
   - Create a registry mapping task names to appropriate test fixtures.
   - Allow modules to request specific dummy data formats.

## 6. Long-term Recommendations

1. **Standardize Evaluation Interfaces:**
   - Establish a consistent interface for all evaluation modules.
   - Create base classes or protocols for different evaluation types (classification, summarization, QA, etc.).

2. **Automated Test Data Generation:**
   - Develop a system to generate realistic synthetic data for each task.
   - Consider using small real datasets for testing that can be bundled with the code.

3. **Metrics Library:**
   - Create a shared metrics library with consistent error handling.
   - Ensure metrics can gracefully handle edge cases like empty inputs.

---

This document tracks evaluation issues as of May 14, 2025. Additional issues may emerge as the codebase evolves.