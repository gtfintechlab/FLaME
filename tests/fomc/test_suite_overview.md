# FOMC Comprehensive Test Suite Overview

## Test Coverage Matrix

| Test Type | Purpose | Frequency | Duration | Coverage Target |
|-----------|---------|-----------|----------|-----------------|
| **Unit Tests** | Test individual functions | Every commit | <1s | >90% |
| **Integration Tests** | Test component interactions | Every PR | <30s | >80% |
| **Smoke Tests** | Quick sanity check | Every deployment | <10s | Critical paths |
| **E2E Tests** | Full workflow validation | Release cycle | <5min | All user flows |
| **Performance Tests** | Benchmark and optimization | Weekly | <10min | Key operations |
| **A/B Tests** | Implementation comparison | Migration phase | <2min | Output parity |

## Test Pyramid

```
        /\
       /E2E\      <- 5% (Comprehensive but slow)
      /------\
     /Integration\ <- 20% (Component interaction)
    /------------\
   /   Smoke     \ <- 10% (Quick validation)
  /--------------\
 /     Unit      \ <- 65% (Fast and focused)
/________________\
```

## Test Suite Structure

```
tests/
├── fomc/
│   ├── unit/
│   │   ├── test_prompt_generation.py
│   │   ├── test_label_extraction.py
│   │   ├── test_response_parsing.py
│   │   ├── test_data_validation.py
│   │   └── test_utils.py
│   │
│   ├── integration/
│   │   ├── test_native_workflow.py
│   │   ├── test_benchforge_workflow.py
│   │   ├── test_data_pipeline.py
│   │   └── test_compatibility.py
│   │
│   ├── smoke/
│   │   ├── test_smoke_native.py
│   │   ├── test_smoke_benchforge.py
│   │   └── test_smoke_parity.py
│   │
│   ├── e2e/
│   │   ├── test_e2e_inference.py
│   │   ├── test_e2e_evaluation.py
│   │   ├── test_e2e_full_pipeline.py
│   │   └── test_e2e_live_api.py
│   │
│   ├── performance/
│   │   ├── test_perf_benchmark.py
│   │   ├── test_perf_comparison.py
│   │   └── test_perf_regression.py
│   │
│   ├── migration/
│   │   ├── test_ab_comparison.py
│   │   ├── test_feature_flags.py
│   │   └── test_rollback.py
│   │
│   └── fixtures/
│       ├── sample_data.py
│       ├── mock_responses.py
│       └── test_config.py
```

## Coverage Requirements

### Unit Tests (>90% coverage)
- All prompt generation functions
- Label extraction logic
- Response parsing utilities
- Data validation functions
- Error handling paths

### Integration Tests (>80% coverage)
- Native FLAME workflow
- BenchForge workflow
- Data pipeline integration
- Cross-implementation compatibility

### Smoke Tests (Critical paths)
- Basic inference capability
- Essential extraction functions
- Output generation
- Error reporting

### E2E Tests (All user flows)
- Complete inference pipeline
- Full evaluation workflow
- Real API interactions
- Production scenarios

## Test Execution Strategy

### Local Development
```bash
# Quick smoke tests
pytest tests/fomc/smoke/ -v

# Unit tests with coverage
pytest tests/fomc/unit/ --cov=flame.fomc --cov-report=html

# Integration tests
pytest tests/fomc/integration/ -v
```

### CI/CD Pipeline
```bash
# Pre-commit: Unit + Smoke
pytest tests/fomc/unit/ tests/fomc/smoke/ --fail-fast

# PR validation: Unit + Integration + Smoke
pytest tests/fomc/unit/ tests/fomc/integration/ tests/fomc/smoke/

# Release: Full suite
pytest tests/fomc/ --cov=flame.fomc --cov-report=xml
```

### Migration Validation
```bash
# A/B testing
pytest tests/fomc/migration/test_ab_comparison.py -v

# Performance comparison
pytest tests/fomc/performance/test_perf_comparison.py --benchmark

# Live API validation (limited samples)
pytest tests/fomc/e2e/test_e2e_live_api.py --samples=5
```

## Success Criteria

1. **Unit Tests**: >90% code coverage, <1s execution
2. **Integration Tests**: All workflows pass, <30s execution
3. **Smoke Tests**: 100% pass rate, <10s execution
4. **E2E Tests**: >95% consistency between implementations
5. **Performance Tests**: <10% regression tolerance
6. **A/B Tests**: >95% output parity

## Test Data Management

### Fixtures
- Sample FOMC statements (10-100 samples)
- Mock LLM responses for each sample
- Expected extraction results
- Edge cases and error scenarios

### Live API Testing
- Limited samples (5-10) to control costs
- Cached responses for regression testing
- Configurable model endpoints
- Rate limiting and retry logic

## Monitoring & Reporting

### Metrics Tracked
- Test execution time
- Coverage percentages
- Failure rates by test type
- Performance benchmarks
- API costs (for live tests)

### Reporting Tools
- pytest-html for HTML reports
- coverage.py for coverage reports
- pytest-benchmark for performance
- Custom migration metrics dashboard