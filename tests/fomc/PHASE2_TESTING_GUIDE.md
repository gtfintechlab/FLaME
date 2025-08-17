# Phase 2 Testing Guide: FOMC Migration

## Overview

This guide provides comprehensive testing coverage for the FOMC migration from native FLAME to BenchForge. The test suite ensures feature parity, performance benchmarks, and production readiness.

## Test Suite Architecture

```
tests/fomc/
├── unit/                      # Core function tests
│   ├── test_prompt_generation.py
│   └── test_label_extraction.py
├── integration/               # Workflow tests
│   └── test_integration_workflow.py
├── smoke/                     # Quick validation
│   └── test_smoke_fomc.py
├── e2e/                       # Live API tests
│   └── test_e2e_live_api.py
├── performance/               # Benchmarking
│   └── test_perf_benchmark.py
├── run_all_tests.py          # Master test runner
├── test_suite_overview.md    # Test documentation
└── PHASE2_TESTING_GUIDE.md   # This guide
```

## Test Coverage Matrix

| Test Level | Files | Tests | Coverage | Duration | Purpose |
|------------|-------|-------|----------|----------|---------|
| **Unit** | 2 | 20+ | >90% | <1s | Function validation |
| **Integration** | 1 | 15+ | >80% | <30s | Component interaction |
| **Smoke** | 1 | 10+ | Critical | <10s | Quick sanity check |
| **E2E** | 1 | 8+ | Full | <5min | Live API validation |
| **Performance** | 1 | 10+ | Key ops | <10min | Benchmarking |

## Quick Start

### 1. Run Smoke Tests (Quick Validation)
```bash
# Fast sanity check (<10 seconds)
uv run pytest tests/fomc/smoke/ -v
```

### 2. Run Unit Tests
```bash
# Core function tests with coverage
uv run pytest tests/fomc/unit/ --cov=flame.code.fomc
```

### 3. Run Integration Tests
```bash
# Component interaction tests
uv run pytest tests/fomc/integration/ -v
```

### 4. Run Performance Tests
```bash
# Benchmark both implementations
uv run pytest tests/fomc/performance/ -v
```

### 5. Run E2E Tests (With API)
```bash
# Live API tests (requires API key)
export TOGETHER_API_KEY=your_key_here
SKIP_LIVE_TESTS=false uv run pytest tests/fomc/e2e/ --samples=5
```

### 6. Run Complete Suite
```bash
# All tests except E2E
uv run python tests/fomc/run_all_tests.py

# Include E2E tests
uv run python tests/fomc/run_all_tests.py --include-live
```

## Test Scenarios

### Unit Tests
- ✅ Prompt generation correctness
- ✅ Label extraction accuracy
- ✅ Response parsing logic
- ✅ Error handling paths
- ✅ Performance characteristics

### Integration Tests
- ✅ Native FLAME workflow
- ✅ BenchForge workflow
- ✅ Data pipeline integration
- ✅ Cross-implementation compatibility
- ✅ Feature flag system

### Smoke Tests
- ✅ Import validation
- ✅ Basic functionality
- ✅ Critical path execution
- ✅ Performance baseline
- ✅ Parity checks

### E2E Tests
- ✅ Real API inference
- ✅ Implementation comparison
- ✅ Error handling
- ✅ Rate limiting
- ✅ Cost tracking

### Performance Tests
- ✅ Prompt generation speed
- ✅ Extraction performance
- ✅ Memory usage
- ✅ Scalability testing
- ✅ Implementation comparison

## Migration Validation

### Pre-Migration Checklist
- [ ] All smoke tests pass
- [ ] Unit test coverage >90%
- [ ] Integration tests pass
- [ ] Performance within 2x of native
- [ ] E2E tests validate with real API

### A/B Testing
```bash
# Run both implementations and compare
uv run python tests/migration/ab_test_fomc.py
```

### Feature Flag Testing
```bash
# Test with BenchForge enabled
export USE_BENCHFORGE_FOMC=1
uv run python main.py --mode inference --task fomc --num_samples 5

# Test with native forced
uv run python main.py --mode inference --task fomc --use-native --num_samples 5
```

## Performance Benchmarks

### Target Metrics
- **Prompt Generation**: <1ms average, <10ms max
- **Label Extraction**: <0.1ms average, <1ms max
- **Memory Usage**: <50MB increase for 1000 operations
- **Scalability**: Linear scaling up to 10,000 operations
- **BenchForge vs Native**: Within 2x performance

### Benchmark Commands
```bash
# Quick benchmark
uv run pytest tests/fomc/performance/test_perf_benchmark.py::TestPromptGenerationPerformance -v

# Full benchmark suite
uv run pytest tests/fomc/performance/ --benchmark-only
```

## CI/CD Integration

### GitHub Actions Workflow
```yaml
name: FOMC Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -r requirements.txt
          uv pip install -e .
      
      - name: Run smoke tests
        run: uv run pytest tests/fomc/smoke/ -v
      
      - name: Run unit tests with coverage
        run: uv run pytest tests/fomc/unit/ --cov=flame.code.fomc --cov-report=xml
      
      - name: Run integration tests
        run: uv run pytest tests/fomc/integration/ -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: fomc-smoke-tests
        name: FOMC Smoke Tests
        entry: uv run pytest tests/fomc/smoke/
        language: system
        pass_filenames: false
        always_run: true
```

## Monitoring & Reporting

### Test Reports
Test results are saved to `test_reports/` directory:
```
test_reports/
├── fomc_test_report_20250816_143022.json
├── performance_benchmark_20250816.json
└── ab_comparison_results.json
```

### Metrics Dashboard
Key metrics to track:
- Test pass rate by suite
- Performance regression detection
- Coverage trends
- E2E success rate
- API cost tracking

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure proper installation
   uv pip install -e .
   uv pip install -r requirements.txt
   ```

2. **API Key Missing**
   ```bash
   export TOGETHER_API_KEY=your_key_here
   # Or skip E2E tests
   export SKIP_LIVE_TESTS=true
   ```

3. **BenchForge Not Available**
   ```bash
   # Install BenchForge
   cd benchforge
   uv pip install -e .
   ```

4. **Test Timeouts**
   ```bash
   # Increase timeout for slow systems
   pytest --timeout=300 tests/fomc/
   ```

## Next Steps

### Week 1 Tasks
- [x] Create comprehensive test suite
- [x] Establish performance baselines
- [ ] Run E2E validation with production data
- [ ] Complete A/B testing

### Week 2 Tasks
- [ ] Enable in development environment
- [ ] Monitor test metrics
- [ ] Address any failures
- [ ] Optimize based on benchmarks

### Week 3 Tasks
- [ ] Production validation
- [ ] Full regression testing
- [ ] Update documentation
- [ ] Team training

### Week 4 Tasks
- [ ] Performance tuning
- [ ] Apply test framework to next task (FPB)
- [ ] Document lessons learned
- [ ] Archive test artifacts

## Success Criteria

✅ **Phase 2 is ready when:**
- All test suites pass
- Coverage >90% for unit tests
- Performance within 2x of native
- E2E tests validate with real API
- A/B tests show >95% consistency
- Documentation complete
- CI/CD pipeline configured

---

**Last Updated**: August 16, 2025
**Status**: Ready for Phase 2 Migration Testing