# Phase 2 Migration Plan: FOMC to BenchForge

## Executive Summary

With Phase 1 validation complete and feature parity confirmed, this document outlines the migration strategy for transitioning FOMC (and subsequently all FLAME tasks) from native implementation to BenchForge.

## Migration Principles

1. **Zero Downtime**: Users can continue using FLAME without interruption
2. **Gradual Rollout**: Progressive migration with rollback capability
3. **Backward Compatibility**: Existing scripts and workflows continue working
4. **Data Continuity**: Results remain compatible across implementations

## Comprehensive Test Strategy

### Test Coverage Matrix
| Test Type | Coverage | Execution Time | Purpose |
|-----------|----------|----------------|---------|
| **Unit Tests** | >90% | <1s | Individual function validation |
| **Integration Tests** | >80% | <30s | Component interaction testing |
| **Smoke Tests** | Critical paths | <10s | Quick sanity checks |
| **E2E Tests** | All workflows | <5min | Full pipeline validation |
| **Performance Tests** | Key operations | <10min | Benchmark & regression detection |

### Test Suite Structure
```
tests/fomc/
├── unit/                 # Core function tests
├── integration/          # Workflow tests
├── smoke/               # Quick validation
├── e2e/                 # Live API tests
├── performance/         # Benchmarking
└── migration/           # A/B comparison
```

### Test Execution Commands
```bash
# Quick validation (smoke tests)
pytest tests/fomc/smoke/ -v

# Full test suite
pytest tests/fomc/ --cov=flame.fomc

# Live API tests (limited samples)
SKIP_LIVE_TESTS=false pytest tests/fomc/e2e/ --samples=5

# Performance benchmarking
pytest tests/fomc/performance/ --benchmark
```

## Phase 2 Timeline

### Week 1: Foundation & Testing
- ✅ Set up comprehensive test suite (unit, integration, smoke, E2E)
- ✅ Create performance benchmarking framework
- Run parallel testing with real data using E2E tests
- Validate output compatibility with A/B testing
- Execute full test suite for both implementations

### Week 2: Gradual Rollout
- Enable BenchForge for development environment
- Monitor performance using benchmarking tests
- Run continuous integration tests
- Collect feedback from test metrics
- Address any failing tests or performance regressions

### Week 3: Production Migration
- Run final E2E validation with production data
- Enable for all users with rollback option
- Monitor test suite health metrics
- Remove duplicate code after test confirmation
- Update documentation with test results

### Week 4: Optimization & Expansion
- Performance tuning based on benchmark results
- Begin migration of next task (FPB) using same test framework
- Document lessons learned and test patterns
- Archive Phase 2 test artifacts

## Implementation Steps

### Step 1: Feature Flag System

Create feature flag configuration:

```python
# src/flame/config.py
import os
from typing import Dict, Set

class MigrationConfig:
    """Configuration for BenchForge migration."""
    
    # Feature flags for each task
    BENCHFORGE_TASKS: Set[str] = set()
    
    # Enable from environment
    if os.getenv('USE_BENCHFORGE_ALL'):
        BENCHFORGE_TASKS = {'all'}
    else:
        # Enable specific tasks
        if os.getenv('USE_BENCHFORGE_FOMC'):
            BENCHFORGE_TASKS.add('fomc')
        if os.getenv('USE_BENCHFORGE_FPB'):
            BENCHFORGE_TASKS.add('fpb')
    
    @classmethod
    def use_benchforge(cls, task: str) -> bool:
        """Check if task should use BenchForge."""
        return 'all' in cls.BENCHFORGE_TASKS or task in cls.BENCHFORGE_TASKS

MIGRATION_CONFIG = MigrationConfig()
```

### Step 2: Update Main Entry Point

Modify `main.py` to route based on feature flags:

```python
# main.py
import argparse
from flame.config import MIGRATION_CONFIG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['inference', 'evaluate'])
    parser.add_argument('--task', type=str)
    parser.add_argument('--use-benchforge', action='store_true', 
                       help='Force BenchForge implementation')
    parser.add_argument('--use-native', action='store_true',
                       help='Force native implementation')
    args = parser.parse_args()
    
    # Determine which implementation to use
    if args.use_benchforge or (not args.use_native and 
                               MIGRATION_CONFIG.use_benchforge(args.task)):
        from flame.main_benchforge import main as benchforge_main
        return benchforge_main()
    else:
        # Use native implementation
        if args.mode == 'inference':
            from flame.code.inference import main as inference_main
            return inference_main(args)
        else:
            from flame.code.evaluate import main as evaluate_main
            return evaluate_main(args)

if __name__ == '__main__':
    main()
```

### Step 3: Create Compatibility Layer

Ensure result compatibility between implementations:

```python
# src/flame/utils/migration_utils.py
import pandas as pd
from pathlib import Path
from typing import Union

def normalize_results(df: pd.DataFrame, source: str = 'unknown') -> pd.DataFrame:
    """Normalize results from either implementation.
    
    Args:
        df: Results DataFrame
        source: 'native' or 'benchforge'
        
    Returns:
        Normalized DataFrame with consistent columns
    """
    # Ensure required columns exist
    required_columns = ['llm_responses', 'extracted_labels', 'actual_labels']
    
    # Map BenchForge columns to FLAME columns if needed
    if source == 'benchforge':
        column_mapping = {
            'raw_response': 'llm_responses',
            'extracted_response': 'extracted_labels',
            'ground_truth': 'actual_labels'
        }
        df = df.rename(columns=column_mapping)
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    return df

def convert_benchforge_to_flame(benchforge_result) -> pd.DataFrame:
    """Convert BenchForge InferenceResult to FLAME format.
    
    Args:
        benchforge_result: BenchForge InferenceResult object
        
    Returns:
        FLAME-compatible DataFrame
    """
    df = benchforge_result.results_df.copy()
    return normalize_results(df, source='benchforge')
```

### Step 4: Parallel Testing Script

Create script for A/B testing:

```python
# tests/migration/ab_test_fomc.py
#!/usr/bin/env python3
"""
A/B Testing for FOMC Implementation Migration
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json

def run_inference(implementation: str, num_samples: int = 10) -> Path:
    """Run inference with specified implementation."""
    
    if implementation == 'native':
        cmd = [
            'uv', 'run', 'python', 'main.py',
            '--mode', 'inference',
            '--task', 'fomc',
            '--num_samples', str(num_samples),
            '--use-native'
        ]
    else:
        cmd = [
            'uv', 'run', 'python', 'main.py',
            '--mode', 'inference', 
            '--task', 'fomc',
            '--num_samples', str(num_samples),
            '--use-benchforge'
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output to get results file path
    for line in result.stdout.split('\n'):
        if 'Results saved to' in line:
            path = line.split('Results saved to')[-1].strip()
            return Path(path)
    
    raise RuntimeError(f"Failed to run {implementation} inference")

def compare_results(native_path: Path, benchforge_path: Path) -> dict:
    """Compare results from both implementations."""
    
    native_df = pd.read_csv(native_path)
    benchforge_df = pd.read_csv(benchforge_path)
    
    # Normalize column names
    from flame.utils.migration_utils import normalize_results
    native_df = normalize_results(native_df, 'native')
    benchforge_df = normalize_results(benchforge_df, 'benchforge')
    
    # Compare key metrics
    comparison = {
        'num_samples': len(native_df),
        'extraction_match_rate': 0.0,
        'response_similarity': 0.0,
        'differences': []
    }
    
    # Compare extracted labels
    if 'extracted_labels' in native_df and 'extracted_labels' in benchforge_df:
        matches = (native_df['extracted_labels'] == benchforge_df['extracted_labels']).sum()
        comparison['extraction_match_rate'] = matches / len(native_df)
    
    # Find differences
    for i in range(len(native_df)):
        if native_df.iloc[i]['extracted_labels'] != benchforge_df.iloc[i]['extracted_labels']:
            comparison['differences'].append({
                'index': i,
                'native': native_df.iloc[i]['extracted_labels'],
                'benchforge': benchforge_df.iloc[i]['extracted_labels']
            })
    
    return comparison

def main():
    """Run A/B test."""
    print("Running A/B Test for FOMC Implementation")
    print("=" * 50)
    
    # Run both implementations
    print("Running native implementation...")
    native_results = run_inference('native', num_samples=20)
    
    print("Running BenchForge implementation...")
    benchforge_results = run_inference('benchforge', num_samples=20)
    
    # Compare results
    print("Comparing results...")
    comparison = compare_results(native_results, benchforge_results)
    
    # Print results
    print("\n" + "=" * 50)
    print("A/B TEST RESULTS")
    print("=" * 50)
    print(f"Samples tested: {comparison['num_samples']}")
    print(f"Extraction match rate: {comparison['extraction_match_rate']:.2%}")
    
    if comparison['extraction_match_rate'] >= 0.95:
        print("✅ Implementations are producing consistent results!")
    else:
        print("⚠️ Differences detected:")
        for diff in comparison['differences'][:5]:
            print(f"  Sample {diff['index']}: native={diff['native']}, benchforge={diff['benchforge']}")
    
    # Save comparison report
    report_path = Path('ab_test_results.json')
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == '__main__':
    main()
```

### Step 5: Monitoring & Rollback

Create monitoring for the migration:

```python
# src/flame/utils/migration_monitor.py
import logging
from datetime import datetime
from typing import Dict, Any
import json
from pathlib import Path

class MigrationMonitor:
    """Monitor BenchForge migration metrics."""
    
    def __init__(self, log_dir: Path = Path('migration_logs')):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.metrics = {
            'native_calls': 0,
            'benchforge_calls': 0,
            'native_errors': 0,
            'benchforge_errors': 0,
            'native_latency': [],
            'benchforge_latency': []
        }
    
    def log_call(self, implementation: str, task: str, 
                 success: bool, latency: float):
        """Log a call to either implementation."""
        
        if implementation == 'native':
            self.metrics['native_calls'] += 1
            self.metrics['native_latency'].append(latency)
            if not success:
                self.metrics['native_errors'] += 1
        else:
            self.metrics['benchforge_calls'] += 1
            self.metrics['benchforge_latency'].append(latency)
            if not success:
                self.metrics['benchforge_errors'] += 1
        
        # Log to file
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'implementation': implementation,
            'task': task,
            'success': success,
            'latency': latency
        }
        
        log_file = self.log_dir / f"migration_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        import numpy as np
        
        summary = self.metrics.copy()
        
        # Calculate averages
        if self.metrics['native_latency']:
            summary['native_avg_latency'] = np.mean(self.metrics['native_latency'])
        if self.metrics['benchforge_latency']:
            summary['benchforge_avg_latency'] = np.mean(self.metrics['benchforge_latency'])
        
        # Calculate error rates
        if self.metrics['native_calls'] > 0:
            summary['native_error_rate'] = (
                self.metrics['native_errors'] / self.metrics['native_calls']
            )
        if self.metrics['benchforge_calls'] > 0:
            summary['benchforge_error_rate'] = (
                self.metrics['benchforge_errors'] / self.metrics['benchforge_calls']
            )
        
        return summary
    
    def should_rollback(self) -> bool:
        """Check if we should rollback to native."""
        
        # Rollback if BenchForge error rate > 5%
        if self.metrics['benchforge_calls'] >= 100:
            error_rate = self.metrics['benchforge_errors'] / self.metrics['benchforge_calls']
            if error_rate > 0.05:
                return True
        
        # Rollback if BenchForge is >2x slower
        if (len(self.metrics['benchforge_latency']) >= 10 and 
            len(self.metrics['native_latency']) >= 10):
            
            bf_avg = np.mean(self.metrics['benchforge_latency'])
            native_avg = np.mean(self.metrics['native_latency'])
            
            if bf_avg > 2 * native_avg:
                return True
        
        return False

# Global monitor instance
MIGRATION_MONITOR = MigrationMonitor()
```

### Step 6: Cutover Script

Final migration script:

```bash
#!/bin/bash
# scripts/migrate_to_benchforge.sh

echo "FLAME to BenchForge Migration"
echo "============================="

# Step 1: Run validation
echo "Step 1: Running validation tests..."
python tests/validation/demo_both_implementations.py
if [ $? -ne 0 ]; then
    echo "❌ Validation failed. Aborting migration."
    exit 1
fi

# Step 2: Enable feature flag
echo "Step 2: Enabling BenchForge for FOMC..."
export USE_BENCHFORGE_FOMC=1

# Step 3: Run A/B test
echo "Step 3: Running A/B test with real data..."
python tests/migration/ab_test_fomc.py

# Step 4: Check results
echo "Step 4: Checking migration metrics..."
python -c "
from flame.utils.migration_monitor import MIGRATION_MONITOR
metrics = MIGRATION_MONITOR.get_metrics()
if MIGRATION_MONITOR.should_rollback():
    print('❌ Rollback recommended based on metrics')
    exit(1)
else:
    print('✅ Migration metrics look good')
"

# Step 5: Update configuration
echo "Step 5: Updating configuration..."
cat > .env.migration << EOF
# BenchForge Migration Settings
USE_BENCHFORGE_FOMC=1
# Add more tasks as they're migrated
# USE_BENCHFORGE_FPB=1
# USE_BENCHFORGE_ALL=1
EOF

echo "✅ Migration complete for FOMC!"
echo ""
echo "Next steps:"
echo "1. Monitor logs in migration_logs/"
echo "2. Run production workloads"
echo "3. If stable after 1 week, remove native implementation"
```

## Rollback Procedure

If issues arise, rollback is simple:

```bash
# Immediate rollback
unset USE_BENCHFORGE_FOMC

# Or force native implementation
export FORCE_NATIVE_IMPLEMENTATION=1
```

## Test Validation Requirements

### Pre-Migration Checklist
- [ ] All unit tests passing (>90% coverage)
- [ ] All integration tests passing
- [ ] Smoke tests complete in <10 seconds
- [ ] E2E tests validated with real API (5+ samples)
- [ ] Performance benchmarks established
- [ ] A/B comparison tests passing

### Continuous Validation
- Run smoke tests on every commit
- Run full test suite on every PR
- Execute E2E tests before deployment
- Monitor performance metrics continuously
- Track test failure rates and trends

## Success Criteria

Phase 2 is complete when:

- [ ] **Test Coverage**: >90% unit test coverage achieved
- [ ] **Integration**: All integration tests passing
- [ ] **E2E Validation**: Live API tests show correct behavior
- [ ] **Performance**: Within 2x of native implementation
- [ ] **A/B Tests**: >95% output consistency
- [ ] **Error Rate**: <1% in production
- [ ] **Documentation**: Test guides and results documented
- [ ] **CI/CD**: Automated test pipeline configured

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Performance regression | Monitor latency, rollback if >2x slower |
| Output inconsistency | A/B testing, gradual rollout |
| Breaking changes | Feature flags, backward compatibility layer |
| User confusion | Clear documentation, training |
| Data loss | All results saved, compatible format |

## Communication Plan

### Week 1
- Internal announcement to dev team
- Enable in development environment

### Week 2
- Documentation updates
- User guide for new features

### Week 3
- Announcement to all users
- Migration guide published

### Week 4
- Deprecation notice for native implementation
- Timeline for removal

## Conclusion

With Phase 1 validation complete, Phase 2 provides a safe, gradual migration path from native FLAME to BenchForge. The feature flag system ensures zero downtime, while comprehensive monitoring allows for quick rollback if needed.

The migration preserves all existing functionality while adding:
- Better error handling
- Improved performance
- Enhanced extensibility
- Professional logging
- Standardized interfaces

Once FOMC migration is successful, the same process can be applied to all other FLAME tasks.