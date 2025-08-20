# FLAME to BenchForge Migration TODO List

## 🎯 Migration Progress Dashboard

**Overall Progress**: 13/23 tasks (57%)
- ✅ **Completed & Tested**: 12 tasks
- 🔧 **Currently Implementing**: 1 task (Headlines)
- 📋 **Real Tasks Remaining**: 10 tasks

---

## ✅ COMPLETED TASKS (12/23)

### Sentiment Analysis
- [x] **FOMC** - Federal Reserve sentiment classification
  - Location: `benchforge/bench_forge/flame/tasks/fomc.py`
  - Status: Production ready, tested
  
- [x] **FPB** - Financial Phrase Bank sentiment
  - Location: `benchforge/bench_forge/flame/tasks/fpb.py`
  - Status: Production ready, tested

### Question Answering
- [x] **ConvFinQA** - Conversational Financial QA
  - Location: `benchforge/bench_forge/flame/tasks/convfinqa.py`
  - Status: Production ready

- [x] **FinQA** - Financial Table QA
  - Location: `benchforge/bench_forge/flame/tasks/finqa.py`
  - Status: Production ready

- [x] **EDTSum** - Earnings Document Summarization
  - Location: `benchforge/bench_forge/flame/tasks/edtsum.py`
  - Status: Production ready

### Target Sentiment Analysis
- [x] **FiQA-SA** - Target-specific sentiment scoring
  - Location: `benchforge/bench_forge/flame/tasks/fiqa_sa.py`
  - Status: Production ready, tested

### Named Entity Recognition
- [x] **FiNER** - Financial NER with BIO tagging
  - Location: `benchforge/bench_forge/flame/tasks/finer.py`
  - Status: Production ready, tested

- [x] **FinEntity** - Financial entity classification
  - Location: `benchforge/bench_forge/flame/tasks/finentity.py`
  - Status: Production ready, tested

### Causality Detection
- [x] **CD** - Causal Detection (BIO sequence)
  - Location: `benchforge/bench_forge/flame/tasks/causal_detection.py`
  - Status: Production ready, tested

- [x] **SC** - Sentence Causality (0-2 classification)
  - Location: `benchforge/bench_forge/flame/tasks/causal_classification.py`
  - Status: Production ready, tested

### Numerical Claims
- [x] **NCC** - Numerical Claim Classification
  - Location: `benchforge/bench_forge/flame/tasks/numclaim.py`
  - Status: Production ready, tested

---

## 🔧 CURRENTLY IMPLEMENTING (1/23)

### News Classification  
- [ ] **Headlines** - Multi-attribute news classification
  - Location: `benchforge/bench_forge/flame/tasks/headlines.py` (in progress)
  - Status: Creating BenchForge implementation with 7 binary attributes
  - TODO:
    - [ ] Complete implementation with exact FLAME prompt
    - [ ] Implement 7-attribute JSON extraction
    - [ ] Test multi-attribute extraction strategies
    - [ ] Validate against FLAME baseline

---

## 📋 REAL TASKS REMAINING (10/23)

### High Priority - Common Financial Tasks

#### 1. **TATQA** - Table and Text QA
**Complexity**: High | **Template**: FinQA pattern
- [ ] Create `benchforge/bench_forge/flame/tasks/tatqa.py`
- [ ] Implement table+text reasoning with arithmetic
- [ ] Port exact FLAME prompt from `src/flame/code/tatqa/`
- [ ] Add table parsing and numeric extraction
- [ ] Test arithmetic operation support
- [ ] Validate exact match and F1 scores

#### 2. **Banking77** - Banking Intent Classification
**Complexity**: Medium | **Template**: Multi-class classification
- [ ] Create `benchforge/bench_forge/flame/tasks/banking77.py`
- [ ] Implement 77-class intent classification
- [ ] Port exact FLAME prompt from `src/flame/code/banking77/`
- [ ] Add intent-specific extraction strategies
- [ ] Test all 77 banking intents
- [ ] Validate multi-class metrics

#### 3. **ECTSum** - Earnings Call Summarization
**Complexity**: Medium | **Template**: Summarization task
- [ ] Create `benchforge/bench_forge/flame/tasks/ectsum.py`
- [ ] Implement earnings call summarization
- [ ] Port exact FLAME prompt from `src/flame/code/ectsum/`
- [ ] Add ROUGE evaluation metrics
- [ ] Test summarization quality
- [ ] Validate against FLAME baseline

#### 4. **FinBench** - Financial Benchmark Suite
**Complexity**: Medium | **Template**: Multi-format benchmark
- [ ] Create `benchforge/bench_forge/flame/tasks/finbench.py`
- [ ] Port exact FLAME prompt from `src/flame/code/finbench/`
- [ ] Implement benchmark evaluation logic
- [ ] Add financial domain metrics
- [ ] Test diverse financial scenarios

#### 5. **FinRed** - Financial Relation Extraction
**Complexity**: High | **Template**: Relation extraction
- [ ] Create `benchforge/bench_forge/flame/tasks/finred.py`
- [ ] Port exact FLAME prompt from `src/flame/code/finred/`
- [ ] Implement relation type extraction
- [ ] Add financial relationship patterns
- [ ] Test relation classification accuracy

### Medium Priority - Specialized Tasks

#### 6. **BizBench** - Business Benchmark
**Complexity**: Medium
- [ ] Create `benchforge/bench_forge/flame/tasks/bizbench.py`
- [ ] Port exact FLAME prompt from `src/flame/code/bizbench/`
- [ ] Implement business scenario evaluation

#### 7. **EconLogicQA** - Economic Logic QA
**Complexity**: Medium
- [ ] Create `benchforge/bench_forge/flame/tasks/econlogicqa.py`
- [ ] Port exact FLAME prompt from `src/flame/code/econlogicqa/`
- [ ] Implement economic reasoning patterns

#### 8. **FNXL** - Financial XBRL Processing
**Complexity**: High
- [ ] Create `benchforge/bench_forge/flame/tasks/fnxl.py`
- [ ] Port exact FLAME prompt from `src/flame/code/fnxl/`
- [ ] Implement XBRL tag extraction

### Lower Priority - General Tasks

#### 9. **SubjectiveQA** - Subjective Question Answering
**Complexity**: Low
- [ ] Create `benchforge/bench_forge/flame/tasks/subjectiveqa.py`
- [ ] Port exact FLAME prompt from `src/flame/code/subjectiveqa/`

#### 10. **RefInd** - Reference Finding
**Complexity**: Medium
- [ ] Create `benchforge/bench_forge/flame/tasks/refind.py`
- [ ] Port exact FLAME prompt from `src/flame/code/refind/`

**Note**: MMLU task exists but is very large and may require special handling

---

## 🚀 Phase 4: INTEGRATION & VALIDATION (Week 8)

### System Integration
- [ ] **Task Registry Updates**
  - [ ] Register all new tasks in `benchforge/bench_forge/flame/registry.py`
  - [ ] Verify auto-registration works
  - [ ] Test task discovery mechanism
  - [ ] Document task metadata

- [ ] **Feature Flag Configuration**
  - [ ] Add feature flags for each task
  - [ ] Configure gradual rollout percentages
  - [ ] Test flag-based routing
  - [ ] Document flag usage

- [ ] **FLAME Compatibility Layer**
  - [ ] Verify column name compatibility
  - [ ] Test data structure conversions
  - [ ] Validate evaluation metrics match
  - [ ] Ensure backward compatibility

### Performance Validation
- [ ] **Extraction Success Rates**
  - [ ] Target: >95% for all tasks
  - [ ] Document actual rates per task
  - [ ] Identify and fix low-performing tasks
  - [ ] Create extraction report

- [ ] **Processing Performance**
  - [ ] Benchmark: <100ms per sample
  - [ ] Batch: >100 samples/second
  - [ ] Memory: <2GB for 10K samples
  - [ ] Document bottlenecks

- [ ] **A/B Testing**
  - [ ] Run parallel FLAME vs BenchForge
  - [ ] Compare outputs for consistency
  - [ ] Validate metrics match within 1%
  - [ ] Document any discrepancies

### Documentation
- [ ] **User Migration Guide**
  - [ ] Step-by-step migration instructions
  - [ ] Breaking changes documentation
  - [ ] Performance improvement guide
  - [ ] Troubleshooting section

- [ ] **API Documentation**
  - [ ] Document all task interfaces
  - [ ] Provide usage examples
  - [ ] Document configuration options
  - [ ] Create quick reference

- [ ] **Performance Report**
  - [ ] Extraction success rates table
  - [ ] Performance benchmarks
  - [ ] Memory usage analysis
  - [ ] Optimization recommendations

---

## 📊 Quick Reference: Implementation Templates

### Template Selection Guide
```
Binary Classification → Use NCC pattern
Multi-class Classification → Use SC/Banking77 pattern  
Multi-attribute Classification → Use Headlines pattern
Sentiment Analysis → Use FPB/FOMC pattern
QA with Tables → Use FinQA/TATQA pattern
QA without Tables → Use ConvFinQA pattern
NER/Sequence Labeling → Use FiNER/CD pattern
Numeric Extraction → Use FiQA-SA pattern
Relation Extraction → Use FinRed pattern
Summarization → Use ECTSum pattern
```

### File Structure Template
```python
# benchforge/bench_forge/flame/tasks/[task_name].py

from benchforge.bench_forge.flame.base import FLAMETask, FLAMEConfig, flame_task
from benchforge.bench_forge.flame.extraction import ExtractionStrategy

@flame_task("[task_name]")
class [TaskName](FLAMETask):
    def __init__(self):
        super().__init__(FLAMEConfig(
            name="[task_name]",
            huggingface_dataset="gtfintechlab/[dataset]",
            valid_labels=[...],
            extraction_strategy=ExtractionStrategy.MULTI_STRATEGY,
            task_type="[classification|qa|ner|sentiment]"
        ))
    
    def create_prompt(self, sample, format="zero_shot"):
        # Task-specific prompt generation
        pass
    
    def extract_answer(self, response):
        # Multi-strategy extraction with fallbacks
        pass
```

---

## 🎯 Success Criteria

### Per-Task Success Metrics
- [ ] Extraction success rate >95%
- [ ] Performance <100ms per sample
- [ ] Memory usage <200MB per 1K samples
- [ ] Unit test coverage >90%
- [ ] Integration tests passing
- [ ] A/B test variance <1%

### Overall Migration Success
- [ ] All 23 real FLAME tasks migrated (13/23 complete)
- [ ] Zero breaking changes for users
- [ ] Performance parity with FLAME
- [ ] Documentation complete
- [ ] Feature flags configured
- [ ] Rollback plan tested

---

## 📅 Timeline

### Current Phase
- Complete Headlines implementation (in progress)
- Begin high-priority real task implementation

### Next Phase
- Implement TATQA, Banking77, ECTSum (high priority)
- Implement FinBench, FinRed (medium priority)

### Final Phase
- Implement remaining specialized tasks
- Complete integration testing
- Performance optimization
- Documentation and rollout preparation

---

## 🔄 Daily Checklist

### Morning
- [ ] Review overnight test results
- [ ] Check extraction success metrics
- [ ] Address any failed tests

### Development
- [ ] Implement next real FLAME task in priority order
- [ ] Write unit tests alongside code
- [ ] Run local validation tests
- [ ] Update progress tracking

### Evening
- [ ] Commit completed work
- [ ] Update TODO status
- [ ] Document any blockers
- [ ] Plan next day's tasks

---

## 📝 Notes and Blockers

### Current Blockers
- None identified

### Decisions Needed
- Prioritize order of remaining 10 real FLAME tasks
- Determine testing strategy for complex tasks (TATQA, FinRed)
- Decide rollout strategy (task by task vs batch)

### Optimization Opportunities
- Consider caching prompt templates
- Batch extraction strategy execution
- Parallelize multi-attribute extraction
- Pre-compile regex patterns

---

**Last Updated**: Documentation Cleanup Session
**Next Review**: After Headlines implementation completion