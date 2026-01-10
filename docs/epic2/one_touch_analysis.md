# FLaME One-Touch Benchmark Analysis Report

## Epic 2 Goal
Create a comprehensive one-command pipeline wrapper (`flame run all`) for running all benchmarks on a new model with integrated reporting.

GitHub Issue: https://github.com/gtfintechlab/FLaME/issues/88

## Current State Analysis

### Existing Infrastructure (70% Complete)
1. **Multi-Task Foundation**: Already supports running multiple tasks via `--tasks` flag
2. **Task Registry**: Central registry with 21 inference tasks and 20 evaluation tasks  
3. **Sequential Execution**: Tasks run sequentially with error collection
4. **Configuration System**: YAML config and CLI args with override capabilities
5. **Output Management**: Structured output paths with model/provider/task organization
6. **Error Handling**: MultiTaskError aggregates failures for comprehensive reporting
7. **Logging System**: Component-based logging with configurable levels

### Current Capabilities
- **Batch Processing**: Run multiple tasks with single command
- **Task Validation**: Pre-flight validation of task names
- **Progress Tracking**: Per-task logging and time measurement
- **Result Storage**: Organized directory structure for results
- **Partial Failure Handling**: Continue execution even if some tasks fail

## Gaps to Address

### 1. Missing "run all" Command
- No explicit `flame run all` command
- Must manually specify all tasks in `--tasks` list
- No automatic task discovery mechanism

### 2. Limited Reporting
- Results saved as individual CSV files per task
- No unified report generation
- No summary statistics across all tasks
- No performance metrics aggregation

### 3. Evaluation Integration
- Inference and evaluation are separate workflows
- No automatic evaluation after inference
- Manual file path tracking for evaluation

### 4. Progress Visualization
- Basic logging output only
- No progress bars or ETA estimation
- No real-time dashboard for long-running benchmarks

### 5. Result Management
- No result comparison across models
- No historical tracking of benchmarks
- No automatic report generation

## Solution Architecture

### Proposed Command Structure

```bash
# Core one-touch command
flame run all --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"

# With options
flame run all --model <model> --evaluate --report --parallel 4

# Subcommands
flame run inference-all    # Run all inference tasks
flame run evaluate-all      # Run all evaluation tasks  
flame run benchmark         # Full pipeline: inference + evaluation + report
```

### Key Components Needed

1. **Command Parser Enhancement**
   - Add `run` subcommand with `all`, `inference-all`, `evaluate-all`, `benchmark` options
   - Automatic task discovery from registry
   - Smart defaults for common workflows

2. **Pipeline Orchestrator**
   - Coordinate inference → evaluation → reporting
   - Handle dependencies between stages
   - Support parallel execution (future enhancement)

3. **Report Generator**
   - Aggregate results across all tasks
   - Generate markdown/HTML summary reports
   - Include performance metrics and comparisons
   - Track benchmark history

4. **Progress Manager**
   - Real-time progress tracking with ETA
   - Rich console output with progress bars
   - Optional web dashboard (future)

## Implementation Plan

### Phase 1: Core Command Infrastructure (Week 1)

**1.1 Update main.py**
```python
# Add 'run' subcommand with options
run_parser = subparsers.add_parser("run", help="Run benchmarks")
run_subparsers = run_parser.add_subparsers(dest="run_command")
run_subparsers.add_parser("all", help="Run all benchmarks")
run_subparsers.add_parser("inference-all", help="Run all inference") 
run_subparsers.add_parser("evaluate-all", help="Run all evaluation")
run_subparsers.add_parser("benchmark", help="Full pipeline")
```

**1.2 Create benchmark_runner.py**
```python
class BenchmarkRunner:
    def run_all(self, model: str, **kwargs):
        """Execute all benchmarks for a model"""
        # 1. Get all tasks from registry
        # 2. Run inference for each task
        # 3. Optionally run evaluation
        # 4. Generate report
```

**1.3 Task Discovery**
- Enhance `task_registry.py` with `get_all_tasks()` function
- Add task metadata (description, category, estimated runtime)

### Phase 2: Progress & Reporting (Week 2)

**2.1 Progress Tracking**
```python
# Add rich progress bars
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

class ProgressManager:
    def track_benchmarks(self, tasks: list):
        """Display progress for benchmark execution"""
```

**2.2 Report Generator**
```python
# src/flame/reporting/report_generator.py
class ReportGenerator:
    def generate_summary(self, results_dir: Path):
        """Generate markdown/HTML summary report"""
    
    def aggregate_metrics(self, task_results: dict):
        """Aggregate performance metrics across tasks"""
```

**2.3 Result Aggregation**
- Collect all task results into unified DataFrame
- Calculate summary statistics
- Generate comparison tables

### Phase 3: Pipeline Integration (Week 3)

**3.1 Pipeline Orchestrator**
```python
# src/flame/pipeline/orchestrator.py
class PipelineOrchestrator:
    def run_benchmark_pipeline(self, model: str):
        """Complete benchmark pipeline"""
        # 1. Run all inference tasks
        # 2. Collect results
        # 3. Run evaluations
        # 4. Generate report
        # 5. Save to benchmark history
```

**3.2 Evaluation Auto-Runner**
- Automatically locate inference results
- Run appropriate evaluation for each task
- Handle tasks without evaluation gracefully

**3.3 Configuration Templates**
```yaml
# configs/benchmark_all.yaml
mode: benchmark
model: "${FLAME_MODEL}"
tasks: all  # Special keyword for all tasks
report:
  format: ["markdown", "html", "json"]
  include_metrics: true
  compare_baseline: true
```

## Specific Code Changes Required

### File Modifications Map

**1. main.py** (Moderate changes)
- Line ~21: Add run subparser
- Line ~189: Handle run command

**2. New file: src/flame/benchmark.py**
- BenchmarkRunner class
- Task orchestration logic
- Report generation integration

**3. src/flame/task_registry.py** (Minor additions)
- Add TASK_METADATA dictionary
- Add get_all_tasks() function

**4. New file: src/flame/reporting/report_generator.py**
- ReportGenerator class
- Markdown/HTML/JSON report generation
- Metrics aggregation

**5. New file: src/flame/utils/progress_utils.py**
- BenchmarkProgress class
- Rich progress bar integration

### Integration Points

1. **Task Registry** → BenchmarkRunner (task discovery)
2. **Inference/Evaluate** → BenchmarkRunner (execution)
3. **Output Utils** → ReportGenerator (result paths)
4. **Config System** → BenchmarkRunner (parameter management)
5. **Logging** → Progress tracking (status updates)

### Testing Strategy

1. **Unit Tests**
   - Test BenchmarkRunner task discovery
   - Test report generation with mock data
   - Test progress tracking

2. **Integration Tests**
   - Test full pipeline with subset of tasks
   - Test error handling in pipeline
   - Test report aggregation

3. **Smoke Tests**
   - Quick validation with 2-3 tasks
   - Verify command parsing
   - Check output structure

## Implementation Roadmap

### Executive Summary

FLaME is **70% ready** for Epic 2 implementation. The multi-task infrastructure provides a solid foundation, but needs extension for true one-touch operation.

### Effort Estimation

**Total Development Time**: 2-3 weeks
- **Week 1**: Core command infrastructure (40 hours)
- **Week 2**: Progress tracking & reporting (30 hours)
- **Week 3**: Integration & testing (20 hours)

### Quick Wins (Can implement immediately)

1. **Add "all" keyword support** (2 hours)
   ```python
   # In main.py, modify run_tasks()
   if "all" in tasks:
       tasks = list(supported_tasks(mode))
   ```

2. **Basic summary report** (4 hours)
   - Aggregate CSV results into single summary
   - Print statistics to console

3. **List command enhancement** (1 hour)
   - Add task counts and categories
   - Show estimated runtime

### Prioritized Feature List

**P0 - Critical**
- [ ] `flame run all` command
- [ ] Automatic task discovery
- [ ] Basic reporting

**P1 - Important**
- [ ] Progress visualization
- [ ] Evaluation auto-run
- [ ] HTML reports

**P2 - Nice to Have**
- [ ] Parallel execution
- [ ] Web dashboard
- [ ] Historical tracking
- [ ] Model comparison

### Next Steps

1. **Immediate Action**: Implement quick wins for MVP
2. **Phase 1**: Build core `BenchmarkRunner` class
3. **Phase 2**: Add progress tracking with `rich` library
4. **Phase 3**: Implement report generation
5. **Phase 4**: Add tests and documentation

### Key Design Decisions

1. **Keep backwards compatibility** - Existing commands continue to work
2. **Progressive enhancement** - Start simple, add features incrementally  
3. **Modular architecture** - Each component (runner, reporter, progress) is independent
4. **Configuration-driven** - Support both CLI and YAML for all features

### Success Metrics

- **Developer Experience**: One command to run all benchmarks
- **Time Savings**: 90% reduction in manual effort
- **Reliability**: <5% failure rate for full benchmark runs
- **Performance**: Complete all tasks in <4 hours for typical model

### Future Enhancements

- **Cloud Integration**: Run benchmarks on cloud infrastructure
- **Model Hub**: Automatic model discovery from HuggingFace
- **Leaderboard**: Compare results across models
- **CI/CD Integration**: Automated benchmarking in GitHub Actions

---

This analysis provides a clear path from the current state to achieving Epic 2's goals. The existing multi-task infrastructure significantly reduces implementation complexity, making the "one-touch" benchmark runner an achievable near-term goal.