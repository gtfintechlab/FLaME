# FLAME to BenchForge Migration TODO List

## 🎯 Migration Progress Dashboard

**Overall Progress**: 12/24 tasks (50%)
- ✅ **Completed & Tested**: 7 tasks
- 🔧 **Code Ready (Needs Testing)**: 5 tasks  
- 📋 **Not Started**: 12 tasks

---

## ✅ Phase 1: COMPLETED TASKS (7/7)

### Sentiment Analysis
- [x] **FOMC** - Federal Reserve sentiment classification
  - Location: `benchforge/bench_forge/flame/tasks/fomc.py`
  - Status: Production ready, 99.6% extraction rate
  
- [x] **FPB** - Financial Phrase Bank sentiment
  - Location: `benchforge/bench_forge/flame/tasks/fpb.py`
  - Status: Production ready, validated

### News Classification  
- [x] **Headlines** - Multi-attribute news classification
  - Location: `benchforge/bench_forge/flame/tasks/headlines.py`
  - Status: Production ready, 7 binary attributes

### Question Answering
- [x] **ConvFinQA** - Conversational Financial QA
  - Location: `benchforge/bench_forge/flame/tasks/convfinqa.py`
  - Status: Production ready, multi-turn support

- [x] **FinQA** - Financial Table QA
  - Location: `benchforge/bench_forge/flame/tasks/finqa.py`
  - Status: Production ready, table reasoning

- [x] **EDTSum** - Earnings Call QA
  - Location: `benchforge/bench_forge/flame/tasks/edtsum.py`
  - Status: Production ready

---

## 🔧 Phase 2: CODE READY - NEEDS TESTING (5/5)

### Target Sentiment Analysis
- [ ] **FiQA-SA** - Target-specific sentiment scoring
  - Location: `benchforge/bench_forge/flame/tasks/fiqa_sa.py`
  - TODO:
    - [ ] Run extraction validation tests
    - [ ] Verify numeric score extraction (-1.0 to 1.0)
    - [ ] Test regression metrics (MSE, MAE, Pearson)
    - [ ] A/B test against FLAME baseline
    - [ ] Document performance metrics

### Named Entity Recognition
- [ ] **FiNER** - Financial NER with BIO tagging
  - Location: `benchforge/bench_forge/flame/tasks/finer.py`
  - TODO:
    - [ ] Test BIO sequence validation
    - [ ] Verify entity boundary detection
    - [ ] Test all 7 extraction strategies
    - [ ] Validate token-level F1 metrics
    - [ ] Performance benchmark (>95% target)

- [ ] **FinEntity** - Financial entity classification (11 types)
  - Location: `benchforge/bench_forge/flame/tasks/finentity.py`
  - TODO:
    - [ ] Test all 11 entity type classifications
    - [ ] Verify multi-class metrics
    - [ ] Validate extraction strategies
    - [ ] Cross-validate with FLAME outputs
    - [ ] Document entity type mappings

### Causality Detection
- [ ] **CD** - Causal Detection (BIO sequence)
  - Location: `benchforge/bench_forge/flame/tasks/causal_detection.py`
  - TODO:
    - [ ] Test cause-effect sequence extraction
    - [ ] Validate BIO tag constraints
    - [ ] Test sequence evaluation metrics
    - [ ] Benchmark token alignment accuracy
    - [ ] Document extraction patterns

- [ ] **SC** - Sentence Causality (0-2 classification)
  - Location: `benchforge/bench_forge/flame/tasks/causal_classification.py`  
  - TODO:
    - [ ] Test numeric label extraction
    - [ ] Validate multi-class metrics
    - [ ] Test confusion matrix generation
    - [ ] A/B test classification accuracy
    - [ ] Performance optimization if needed

### Numerical Claims
- [ ] **NCC** - Numerical Claim Classification
  - Location: `benchforge/bench_forge/flame/tasks/numclaim.py`
  - TODO:
    - [ ] Test binary classification (0/1)
    - [ ] Validate financial pattern detection
    - [ ] Test extraction strategies
    - [ ] Verify binary metrics (precision, recall, F1)
    - [ ] Document claim patterns

---

## 📋 Phase 3: PENDING IMPLEMENTATION (12/12)

### High Priority - Core Financial Tasks (Week 5)

#### 1. **TSA** - Twitter Sentiment Analysis
**Complexity**: Low | **Template**: FPB pattern
- [ ] Create `benchforge/bench_forge/flame/tasks/tsa.py`
- [ ] Port prompt templates from `flame/tasks/tsa.py`
- [ ] Implement 3-class sentiment (positive/negative/neutral)
- [ ] Add Twitter-specific text preprocessing
- [ ] Test with social media text patterns
- [ ] Validate against FLAME baseline

#### 2. **TATQA** - Table and Text QA
**Complexity**: Medium | **Template**: FinQA pattern
- [ ] Create `benchforge/bench_forge/flame/tasks/tatqa.py`
- [ ] Implement table+text reasoning
- [ ] Port numeric extraction logic
- [ ] Add arithmetic operation support
- [ ] Test table parsing accuracy
- [ ] Validate exact match and F1 scores

#### 3. **FinQABench** - Financial QA Benchmark
**Complexity**: Medium | **Template**: FinQA pattern
- [ ] Create `benchforge/bench_forge/flame/tasks/finqabench.py`
- [ ] Analyze question types and formats
- [ ] Implement comprehensive QA patterns
- [ ] Add benchmark-specific metrics
- [ ] Test on diverse question types
- [ ] Document performance targets

### Medium Priority - Classification Tasks (Week 6)

#### 4. **MA** - Merger & Acquisition Classification
**Complexity**: Low | **Template**: NCC binary pattern
- [ ] Create `benchforge/bench_forge/flame/tasks/ma.py`
- [ ] Find/create dataset configuration
- [ ] Implement binary classification (M&A event: yes/no)
- [ ] Add financial event keywords
- [ ] Test extraction strategies
- [ ] Validate classification metrics

#### 5. **MLESG** - Multi-label ESG Classification
**Complexity**: Medium | **Template**: Headlines multi-attribute
- [ ] Create `benchforge/bench_forge/flame/tasks/mlesg.py`
- [ ] Implement multi-label classification
- [ ] Port ESG category definitions
- [ ] Add per-category metrics
- [ ] Test multi-label extraction
- [ ] Validate against ESG standards

#### 6. **FLS** - Forward-Looking Statements
**Complexity**: Low | **Template**: NCC binary pattern
- [ ] Create `benchforge/bench_forge/flame/tasks/fls.py`
- [ ] Implement binary classification
- [ ] Add temporal language patterns
- [ ] Test forward-looking indicators
- [ ] Validate classification accuracy

#### 7. **NER** - Standard Named Entity Recognition
**Complexity**: Medium | **Template**: FiNER pattern
- [ ] Create `benchforge/bench_forge/flame/tasks/ner.py`
- [ ] Port standard NER tags (PER, ORG, LOC, etc.)
- [ ] Implement BIO tagging
- [ ] Add entity type validation
- [ ] Test boundary detection
- [ ] Validate F1 scores

### Low Priority - Benchmark Suites (Weeks 7-8)

#### 8. **CFA** - CFA Exam Benchmark
**Complexity**: High | **Template**: Multi-format QA
- [ ] Create `benchforge/bench_forge/flame/tasks/cfa.py`
- [ ] Analyze CFA question formats
- [ ] Implement multiple choice support
- [ ] Add professional exam metrics
- [ ] Test on sample CFA questions
- [ ] Document score calculations

#### 9. **FINEVAL** - Comprehensive Financial Evaluation
**Complexity**: High | **Template**: Mixed tasks
- [ ] Create `benchforge/bench_forge/flame/tasks/fineval.py`
- [ ] Implement multi-task evaluation
- [ ] Port diverse question types
- [ ] Add comprehensive metrics
- [ ] Test all subtasks
- [ ] Create unified evaluation

#### 10-15. **Flare Variants** (Enhanced versions)
**Complexity**: Low | **Template**: Base task + enhancements

##### 10. **Flare-FOMC**
- [ ] Create `benchforge/bench_forge/flame/tasks/flare_fomc.py`
- [ ] Extend FOMC with enhanced features
- [ ] Add additional extraction strategies
- [ ] Test enhanced accuracy

##### 11. **Flare-FPB**
- [ ] Create `benchforge/bench_forge/flame/tasks/flare_fpb.py`
- [ ] Extend FPB with improvements
- [ ] Add context-aware features
- [ ] Validate enhancements

##### 12. **Flare-Headlines**
- [ ] Create `benchforge/bench_forge/flame/tasks/flare_headlines.py`
- [ ] Extend Headlines with more attributes
- [ ] Add advanced extraction
- [ ] Test attribute accuracy

##### 13. **Flare-FINQA**
- [ ] Create `benchforge/bench_forge/flame/tasks/flare_finqa.py`
- [ ] Extend FinQA with complex reasoning
- [ ] Add multi-step calculations
- [ ] Validate arithmetic accuracy

##### 14. **Flare-SM** (Stock Movement)
- [ ] Create `benchforge/bench_forge/flame/tasks/flare_sm.py`
- [ ] Implement stock movement prediction
- [ ] Add market indicators
- [ ] Test prediction accuracy

##### 15. **Flare-ECTSUM** (Earnings Call Summarization)
- [ ] Create `benchforge/bench_forge/flame/tasks/flare_ectsum.py`
- [ ] Implement summarization task
- [ ] Add ROUGE metrics
- [ ] Test summary quality

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
Multi-class Classification → Use SC pattern  
Multi-label Classification → Use Headlines pattern
Sentiment Analysis → Use FPB pattern
QA with Tables → Use FinQA pattern
QA without Tables → Use ConvFinQA pattern
NER/Sequence Labeling → Use FiNER pattern
Numeric Extraction → Use FiQA-SA pattern
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
- [ ] All 24 tasks migrated
- [ ] Zero breaking changes for users
- [ ] Performance improvement >20%
- [ ] Documentation complete
- [ ] Feature flags configured
- [ ] Rollback plan tested

---

## 📅 Timeline

### Week 5 (Current)
- Complete testing of 5 code-ready tasks
- Implement TSA, TATQA, FinQABench

### Week 6
- Implement MA, MLESG, FLS, NER
- Begin integration testing

### Week 7
- Implement CFA, FINEVAL
- Implement Flare variants
- Performance optimization

### Week 8
- Complete integration testing
- A/B testing and validation
- Documentation and rollout preparation

---

## 🔄 Daily Checklist

### Morning
- [ ] Review overnight test results
- [ ] Check extraction success metrics
- [ ] Address any failed tests

### Development
- [ ] Implement next task in priority order
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
- Confirm dataset availability for MA, MLESG, FLS tasks
- Decide on Flare variant implementation priority
- Determine rollout strategy (task by task vs batch)

### Optimization Opportunities
- Consider caching prompt templates
- Batch extraction strategy execution
- Parallelize multi-attribute extraction
- Pre-compile regex patterns

---

**Last Updated**: Current Session
**Next Review**: After Phase 2 testing completion