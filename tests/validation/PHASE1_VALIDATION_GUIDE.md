# Phase 1 Validation Guide: FOMC Feature Parity Testing

## Overview

This guide documents the Phase 1 validation process for ensuring FOMC implementation feature parity between native FLAME and BenchForge versions.

## Validation Objectives

1. **Functional Parity**: Both implementations produce identical prompts
2. **Extraction Parity**: Both handle response extraction identically
3. **API Compatibility**: Both expose compatible interfaces
4. **Error Handling**: Both handle edge cases appropriately

## Test Components

### 1. Validation Script (`phase1_fomc_validation.py`)

Comprehensive Python test suite that:
- Tests prompt generation for both implementations
- Validates extraction logic (rule-based and LLM-based)
- Compares outputs for consistency
- Generates detailed reports

### 2. Comparison Runner (`run_fomc_comparison.sh`)

Bash script that:
- Tests basic imports and functionality
- Runs validation suite
- Generates timestamped results
- Provides clear pass/fail status

## Running the Tests

### Quick Test (No API calls needed)

```bash
# Make script executable
chmod +x tests/validation/run_fomc_comparison.sh

# Run validation
./tests/validation/run_fomc_comparison.sh
```

This will:
1. Test native FLAME imports and basic functions
2. Test BenchForge imports and initialization
3. Run comparison tests
4. Generate validation report

### Detailed Test with Custom Samples

```bash
# Run with more samples
python tests/validation/phase1_fomc_validation.py --samples 20 --verbose

# Specify output directory
python tests/validation/phase1_fomc_validation.py --output my_results
```

## Interpreting Results

### Success Criteria

✅ **PASS** when:
- Prompt match rate > 90%
- Extraction match rate > 90%
- No critical errors in either implementation
- Both implementations handle edge cases

⚠️ **REVIEW** when:
- Minor differences in prompt formatting (whitespace, etc.)
- Non-critical warning messages
- Performance differences (not functionality)

❌ **FAIL** when:
- Prompt match rate < 90%
- Extraction logic produces different results
- Critical errors or exceptions
- Missing functionality in either implementation

### Output Files

After running validation, check these files:

1. **`validation_summary_*.md`** - Human-readable summary
2. **`validation_report_*.json`** - Detailed JSON report
3. **`native_test.log`** - Native FLAME test output
4. **`benchforge_test.log`** - BenchForge test output
5. **`validation.log`** - Full validation run log

## Test Coverage

### Prompt Generation
- [x] Zero-shot prompts
- [x] Few-shot prompts
- [x] Chain-of-thought prompts
- [x] Edge cases (empty text, special characters)

### Extraction Logic
- [x] Direct label extraction (e.g., "HAWKISH")
- [x] Contextual extraction (e.g., "The answer is DOVISH")
- [x] Messy responses (multi-line, explanations)
- [x] LLM-based extraction fallback
- [x] Invalid/missing labels

### Label Mapping
- [x] Text to numeric conversion
- [x] Case insensitivity
- [x] Invalid label handling

## Phase 1 Completion Checklist

Before proceeding to Phase 2, verify:

- [ ] Validation script runs without errors
- [ ] Both implementations import successfully
- [ ] Prompt generation matches (>90% similarity)
- [ ] Extraction logic produces same results
- [ ] LLM-based extraction available in BenchForge
- [ ] Label mapping is consistent
- [ ] Error handling is equivalent
- [ ] Performance is comparable

## Next Steps: Phase 2 Migration

Once Phase 1 validation passes:

### 1. Run with Real Data (Small Scale)

```bash
# Test with actual API calls (10 samples)
uv run python main.py --mode inference --task fomc --num_samples 10

# Test with BenchForge
uv run python main_benchforge.py --mode inference --task fomc --num_samples 10
```

### 2. Compare Real Outputs

```python
# Compare CSV outputs
import pandas as pd

native_df = pd.read_csv('results/fomc/fomc_native_*.csv')
benchforge_df = pd.read_csv('results/fomc/fomc_benchforge_*.csv')

# Check if results match
comparison = native_df['extracted_labels'] == benchforge_df['extracted_labels']
match_rate = comparison.mean()
print(f"Real data match rate: {match_rate:.2%}")
```

### 3. Enable Gradual Migration

```python
# In config or environment
USE_BENCHFORGE_FOMC = True  # Feature flag

# In main.py
if USE_BENCHFORGE_FOMC and task == 'fomc':
    return run_benchforge_inference(task, args)
else:
    return run_native_inference(task, args)
```

### 4. Monitor and Validate

- Track performance metrics
- Monitor error rates
- Collect user feedback
- A/B test if needed

## Troubleshooting

### Common Issues

**BenchForge not found**
```bash
# Initialize submodule
git submodule update --init --recursive

# Install BenchForge
cd benchforge
pip install -e .
```

**Import errors**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)/benchforge"
```

**Test data issues**
```python
# Verify test data format
df = pd.read_csv('test_data.csv')
assert 'sentence' in df.columns
assert 'label' in df.columns
```

## Summary

Phase 1 validation ensures that:

1. **Both implementations work** - Can be imported and initialized
2. **Outputs are compatible** - Generate similar prompts and extractions
3. **Features are complete** - All functionality is present
4. **Quality is maintained** - No regression in capabilities

Once validation passes, you're ready for Phase 2: Gradual migration with real data and production deployment.