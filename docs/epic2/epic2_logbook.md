# Epic 2 Implementation Logbook

## Overview
This logbook tracks the implementation of Epic 2: One-Touch Benchmark Runner for FLaME.

**GitHub Issue**: https://github.com/gtfintechlab/FLaME/issues/88  
**Goal**: Create `flame run all` command for comprehensive benchmarking  
**Status**: 🟡 In Progress (Analysis Complete)  
**Started**: 2025-08-12  

## Quick Links
- [One-Touch Analysis Report](./one_touch_analysis.md)
- [Multi-Task Guide](../flame_multi_task_guide.md)
- [Project Structure](../flame_project_structure.md)

---

## Implementation Tracker

### Phase 1: Core Command Infrastructure
- [ ] Add `run` subcommand to main.py
- [ ] Create BenchmarkRunner class
- [ ] Implement task discovery from registry
- [ ] Add "all" keyword support for tasks
- [ ] Write unit tests for command parsing

### Phase 2: Progress & Reporting  
- [ ] Add rich library dependency
- [ ] Create ProgressManager class
- [ ] Implement ReportGenerator
- [ ] Design report templates (markdown/HTML)
- [ ] Add metrics aggregation

### Phase 3: Pipeline Integration
- [ ] Build PipelineOrchestrator
- [ ] Auto-link inference → evaluation
- [ ] Create benchmark history tracking
- [ ] Add configuration templates
- [ ] Integration testing

---

## Development Log

### 2025-08-12: Initial Analysis & Planning
**Time**: 10:00 AM  
**Author**: Claude + gmatlin  
**Status**: ✅ Complete  

**Activities**:
1. Analyzed current FLaME architecture
2. Identified gaps between current state and Epic 2 requirements
3. Designed solution architecture for one-touch command
4. Created implementation plan with specific code changes
5. Estimated effort: 2-3 weeks total development time

**Key Findings**:
- FLaME is 70% ready for Epic 2 implementation
- Multi-task infrastructure provides solid foundation
- Main gaps: command structure, reporting, progress visualization

**Decisions Made**:
- Use subcommand structure: `flame run all`
- Keep backwards compatibility with existing commands
- Start with sequential execution, add parallel later
- Use rich library for progress visualization

**Next Steps**:
1. Implement quick wins (2-7 hours total):
   - Add "all" keyword support
   - Basic summary reporting
   - Enhanced list command
2. Begin Phase 1 core infrastructure

---

## Quick Wins Implementation

### 1. Add "all" keyword support
**Estimated**: 2 hours  
**Actual**: _pending_  
**File**: main.py  
**Status**: ⏳ Not Started  

```python
# Modification needed in run_tasks()
if "all" in tasks:
    tasks = list(supported_tasks(mode))
```

### 2. Basic Summary Report
**Estimated**: 4 hours  
**Actual**: _pending_  
**Status**: ⏳ Not Started  

### 3. List Command Enhancement  
**Estimated**: 1 hour  
**Actual**: _pending_  
**Status**: ⏳ Not Started  

---

## Code Snippets & Examples

### Example: Running All Benchmarks
```bash
# Current approach (manual)
python main.py --mode inference --tasks fomc numclaim finer finentity causal_classification subjectiveqa ectsum fnxl fpb banking77 bizbench causal_detection convfinqa edtsum finbench finqa finred fiqa_task1 fiqa_task2 headlines refind tatqa --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"

# Future approach (one-touch)
flame run all --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
```

### Example: Benchmark Config
```yaml
# configs/benchmark_all.yaml
mode: benchmark
model: "${FLAME_MODEL}"
tasks: all
parallel: 4
report:
  format: ["markdown", "html"]
  include_metrics: true
```

---

## Design Decisions

### 1. Command Structure
**Decision**: Use subcommand pattern `flame run all`  
**Rationale**: Clear, extensible, follows CLI best practices  
**Alternatives Considered**: 
- Single flag like `--run-all` (less flexible)
- Separate script (breaks integration)

### 2. Progress Visualization
**Decision**: Use rich library  
**Rationale**: Modern, feature-rich, good Python integration  
**Alternatives Considered**:
- tqdm (less features)
- Custom implementation (unnecessary complexity)

### 3. Report Format
**Decision**: Support multiple formats (MD, HTML, JSON)  
**Rationale**: Different use cases need different formats  
**Trade-offs**: More complexity but better flexibility

---

## Testing Notes

### Test Scenarios
1. **Happy Path**: Run all tasks successfully
2. **Partial Failure**: Some tasks fail, others succeed
3. **Configuration**: Various config combinations
4. **Performance**: Large model benchmarking
5. **Interruption**: Handle Ctrl+C gracefully

### Test Commands
```bash
# Quick smoke test (2-3 tasks)
pytest tests/integration/test_benchmark_runner.py::test_smoke

# Full integration test
pytest tests/integration/test_benchmark_runner.py::test_full_pipeline

# Performance test
pytest tests/integration/test_benchmark_runner.py::test_performance -v
```

---

## Performance Metrics

### Baseline (Current)
- Manual execution time: ~5 minutes setup per model
- Error recovery: Manual intervention required
- Report generation: Manual aggregation needed

### Target (Epic 2)
- One-touch execution: <30 seconds setup
- Automatic error recovery with reporting
- Automated report generation in multiple formats

---

## Dependencies & Requirements

### New Dependencies
- `rich>=13.0.0` - Progress bars and console output
- `jinja2>=3.0.0` - HTML report templating
- `pandas>=1.5.0` - Already included, for data aggregation

### System Requirements
- Python 3.8+
- 8GB RAM minimum for full benchmark suite
- Network access for API calls

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| API rate limiting | High | High | Implement backoff, batching |
| Memory exhaustion | Medium | High | Add memory monitoring |
| Long runtime | High | Medium | Add checkpointing |
| Task failures | Medium | Low | Error aggregation exists |

---

## Resources & References

### Documentation
- [FLaME Multi-Task Guide](../flame_multi_task_guide.md)
- [Task Registry Implementation](../../src/flame/task_registry.py)
- [Current Main Entry Point](../../main.py)

### External Resources
- [Rich Documentation](https://rich.readthedocs.io/)
- [Click CLI Framework](https://click.palletsprojects.com/) (potential future)
- [Jinja2 Templates](https://jinja.palletsprojects.com/)

---

## Notes & Ideas

### Future Enhancements
- **Parallel Execution**: Use multiprocessing for independent tasks
- **Web Dashboard**: Real-time monitoring via Flask/FastAPI
- **Cloud Integration**: Run on AWS/GCP/Azure
- **Model Comparison**: Side-by-side benchmark results
- **Caching**: Smart caching of completed tasks
- **Resume Capability**: Continue from interruption point

### Optimization Opportunities
- Batch API calls more aggressively
- Pre-compile prompts for all tasks
- Cache model tokenizer initialization
- Stream results to disk during execution

---

## Communication Log

### Stakeholder Updates
- **2025-08-12**: Initial analysis complete, 2-3 week estimate
- _Next update_: After quick wins implementation

### Questions for Team
1. Preference for progress visualization (console vs web)?
2. Priority of parallel execution vs sequential reliability?
3. Report format preferences for different audiences?

---

## Session Notes

### Working Session: 2025-08-12
- Completed comprehensive codebase analysis
- Identified 70% readiness for Epic 2
- Created detailed implementation plan
- Established logbook for tracking progress

### Next Session Goals
1. Implement "all" keyword support
2. Test with 2-3 tasks
3. Begin BenchmarkRunner class structure

---

*This logbook is a living document and will be updated throughout the Epic 2 implementation.*