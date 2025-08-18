# FLAME to BenchForge Complete Migration Plan

## Executive Summary

This document provides a comprehensive migration plan for transitioning all 24 FLAME native tasks to the BenchForge implementation. The migration focuses on leveraging BenchForge's robust infrastructure while preserving FLAME's financial domain expertise.

**Migration Status Overview:**
- ✅ **Completed**: 7 tasks (FOMC, FPB, FiQA-SA, Headlines, ConvFinQA, FinQA, EDTSum)
- 🔧 **Code Ready**: 5 tasks (FiNER, FinEntity, CD, SC, NCC)
- 📋 **Pending**: 12 tasks (TSA, Flare series, benchmarks, others)

## Architecture Principles

### Division of Responsibilities

**FLAME Core Responsibilities:**
- Task-specific prompt templates and domain knowledge
- Financial validation logic
- User-facing CLI and workflows
- Legacy compatibility layer

**BenchForge Engine Responsibilities:**
- LLM client management (via LiteLLM)
- Response extraction strategies (7-strategy system)
- Batch processing and parallelization
- Caching and performance optimization
- Metrics computation and evaluation

### Migration Pattern

```python
# Standard BenchForge Task Pattern
@flame_task("task_name")
class TaskName(FLAMETask):
    def __init__(self):
        super().__init__(FLAMEConfig(
            name="task_name",
            huggingface_dataset="gtfintechlab/dataset_name",
            valid_labels=["LABEL1", "LABEL2"],
            extraction_strategy=ExtractionStrategy.MULTI_STRATEGY,
            task_type="classification|qa|ner|sentiment"
        ))
    
    def create_prompt(self, sample: Dict[str, Any], format: str = "zero_shot") -> str:
        """Create task-specific prompt"""
        return self._format_prompt(sample, format)
    
    def extract_answer(self, response: str) -> Any:
        """Extract answer with multi-strategy fallback"""
        return self._multi_strategy_extract(response)
```

## Task Migration Categories

### Category 1: Sentiment Analysis (Simple - 1 week)
**Complexity**: Low | **Priority**: High | **Business Impact**: High

| Task | Status | Migration Steps |
|------|--------|-----------------|
| FOMC | ✅ Complete | Reference implementation |
| FPB | ✅ Complete | Template for 3-class sentiment |
| FiQA-SA | 🔧 Code Ready | Needs testing and validation |
| TSA | 📋 Pending | Apply FPB pattern with Twitter-specific prompts |

**Migration Template**: Use FPB implementation as base, modify prompts for domain.

### Category 2: Question Answering (Medium - 2 weeks)
**Complexity**: Medium | **Priority**: High | **Business Impact**: High

| Task | Status | Migration Steps |
|------|--------|-----------------|
| ConvFinQA | ✅ Complete | Multi-turn conversation handling |
| FinQA | ✅ Complete | Table reasoning with numeric extraction |
| EDTSum | ✅ Complete | Earnings call QA |
| FinQABench | 📋 Pending | Apply FinQA pattern |
| TATQA | 📋 Pending | Table+text reasoning similar to FinQA |

**Key Challenges**: 
- Multi-turn conversation context
- Table parsing and formatting
- Numeric answer extraction and validation

### Category 3: Named Entity Recognition (Complex - 2 weeks)
**Complexity**: High | **Priority**: Medium | **Business Impact**: Medium

| Task | Status | Migration Steps |
|------|--------|-----------------|
| FiNER | 🔧 Code Ready | BIO tagging with sequence validation |
| FinEntity | 🔧 Code Ready | 11-type entity classification |
| NER | 📋 Pending | Apply FiNER pattern |

**Key Challenges**:
- Token-level alignment
- BIO sequence validation
- Entity boundary detection
- Multi-word entity handling

### Category 4: Classification Tasks (Simple - 1 week)
**Complexity**: Low-Medium | **Priority**: Medium | **Business Impact**: Medium

| Task | Status | Migration Steps |
|------|--------|-----------------|
| Headlines | ✅ Complete | Multi-attribute binary classification |
| CD | 🔧 Code Ready | Causal detection with BIO |
| SC | 🔧 Code Ready | Multi-class causality |
| NCC | 🔧 Code Ready | Binary claim detection |
| MA | 📋 Pending | Binary M&A classification |
| MLESG | 📋 Pending | Multi-label ESG classification |
| FLS | 📋 Pending | Binary forward-looking statements |

**Migration Template**: Use SC implementation for multi-class, NCC for binary.

### Category 5: Benchmark Suites (Medium - 2 weeks)
**Complexity**: Medium | **Priority**: Low | **Business Impact**: Research

| Task | Status | Migration Steps |
|------|--------|-----------------|
| CFA | 📋 Pending | Professional exam format |
| FINEVAL | 📋 Pending | Comprehensive evaluation |
| Flare-FOMC | 📋 Pending | Enhanced FOMC variant |
| Flare-FPB | 📋 Pending | Enhanced FPB variant |
| Flare-Headlines | 📋 Pending | Enhanced Headlines variant |
| Flare-FINQA | 📋 Pending | Enhanced FinQA variant |

**Strategy**: Leverage base task implementations with benchmark-specific modifications.

## Implementation Roadmap

### Phase 1: Core Financial Tasks (Weeks 1-2) ✅ COMPLETE
- [x] FOMC - Federal Reserve sentiment
- [x] FPB - Financial phrase sentiment  
- [x] Headlines - News classification
- [x] Core QA tasks (ConvFinQA, FinQA, EDTSum)

### Phase 2: Entity & Classification (Weeks 3-4) 🔧 IN PROGRESS
- [x] FiNER - Financial NER
- [x] FinEntity - Entity classification
- [x] Causality tasks (CD, SC, NCC)
- [ ] Testing and validation
- [ ] Performance benchmarking

### Phase 3: Remaining Tasks (Weeks 5-6) 📋 PLANNED
- [ ] TSA - Twitter sentiment
- [ ] TATQA - Table QA
- [ ] MA, MLESG, FLS classifications
- [ ] Standard NER tasks

### Phase 4: Benchmark Suites (Weeks 7-8) 📋 PLANNED
- [ ] CFA professional exam
- [ ] FINEVAL comprehensive
- [ ] All Flare variants
- [ ] Final integration testing

## Migration Checklist Per Task

### Pre-Migration Analysis
- [ ] Analyze current FLAME implementation
- [ ] Document prompt templates and formats
- [ ] Identify valid labels and data structure
- [ ] Note evaluation metrics and special logic
- [ ] Assess extraction complexity

### Implementation Steps
1. [ ] Create task file in `benchforge/bench_forge/flame/tasks/`
2. [ ] Implement `FLAMEConfig` with task metadata
3. [ ] Port `create_prompt()` method with all formats
4. [ ] Implement multi-strategy extraction (5-7 strategies)
5. [ ] Add task-specific validation logic
6. [ ] Register task in BenchForge registry

### Validation & Testing
- [ ] Unit tests for prompt generation
- [ ] Extraction strategy tests
- [ ] Integration test with sample data
- [ ] A/B test against FLAME baseline
- [ ] Performance benchmarking
- [ ] Document extraction success rates

### Documentation
- [ ] Update task documentation
- [ ] Add usage examples
- [ ] Document any breaking changes
- [ ] Update migration status

## Technical Implementation Details

### Multi-Strategy Extraction System
```python
EXTRACTION_STRATEGIES = [
    "direct_extract",      # Look for exact label match
    "keyword_search",      # Search for label keywords
    "pattern_match",       # Regex patterns
    "structured_parse",    # Parse JSON/XML structures
    "fuzzy_match",        # Fuzzy string matching
    "semantic_similarity", # Embedding-based matching
    "llm_reextract"       # Use LLM to re-extract
]
```

### FLAME Compatibility Requirements
```python
# Required output columns for FLAME evaluation
OUTPUT_COLUMNS = {
    "sample_index": int,          # Sample ID
    "predicted_answer": Any,       # Model prediction
    "gold_answer": Any,           # Ground truth
    "exact_match": bool,          # Evaluation metric
    "model_name": str,            # Model identifier
    "extraction_success": bool,   # Extraction status
}
```

### Performance Targets
- **Extraction Success Rate**: >95% (vs FLAME ~80%)
- **Processing Speed**: <100ms per sample
- **Batch Efficiency**: 100+ samples/second
- **Memory Usage**: <2GB for 10K samples
- **Cache Hit Rate**: >50% for repeated runs

## Risk Mitigation

### Technical Risks
1. **Extraction Failures**: Mitigated by 7-strategy system with fallbacks
2. **Performance Regression**: Feature flags enable quick rollback
3. **Breaking Changes**: Comprehensive A/B testing before migration
4. **Data Format Changes**: Validation layer ensures compatibility

### Migration Risks
1. **Downtime**: Zero-downtime migration via feature flags
2. **Data Loss**: Complete response storage preserves all information
3. **Compatibility**: FLAME compatibility layer maintains interfaces
4. **User Impact**: Gradual rollout with monitoring

## Success Metrics

### Migration Success Criteria
- [ ] 100% task coverage (24/24 tasks migrated)
- [ ] >95% extraction success rate across all tasks
- [ ] <5% performance regression vs FLAME baseline
- [ ] Zero breaking changes for existing users
- [ ] Complete documentation and examples

### Quality Metrics
- [ ] 100% unit test coverage
- [ ] >90% integration test coverage
- [ ] <1% error rate in production
- [ ] <100ms p95 latency per request
- [ ] >50% code reuse across tasks

## Appendix A: Task File Locations

### Completed Migrations
```
benchforge/bench_forge/flame/tasks/
├── fomc.py          ✅ Federal Reserve sentiment
├── fpb.py           ✅ Financial phrase sentiment
├── fiqa_sa.py       🔧 Target sentiment analysis
├── headlines.py     ✅ News classification
├── convfinqa.py     ✅ Conversation QA
├── finqa.py         ✅ Table reasoning QA
├── edtsum.py        ✅ Earnings call QA
├── finer.py         🔧 Financial NER
├── finentity.py     🔧 Entity classification
├── causal_detection.py  🔧 Causal BIO tagging
├── causal_classification.py 🔧 Causality classification
└── numclaim.py      🔧 Numerical claim detection
```

### Pending Migrations
```
flame/tasks/
├── tsa.py           📋 Twitter sentiment
├── tatqa.py         📋 Table+text QA
├── finqabench.py    📋 QA benchmark
├── ma.py            📋 M&A classification
├── mlesg.py         📋 ESG classification
├── fls.py           📋 Forward-looking statements
├── cfa.py           📋 CFA exam benchmark
├── fineval.py       📋 Comprehensive benchmark
└── flare_*.py       📋 Enhanced task variants
```

## Appendix B: Common Migration Patterns

### Pattern 1: Simple Classification
```python
# For binary/multi-class classification tasks
@flame_task("task_name")
class SimpleClassification(FLAMETask):
    def __init__(self):
        super().__init__(FLAMEConfig(
            name="task_name",
            valid_labels=["CLASS_A", "CLASS_B"],
            extraction_strategy=ExtractionStrategy.KEYWORD
        ))
```

### Pattern 2: Numeric Extraction
```python
# For QA tasks with numeric answers
def extract_numeric(self, text: str) -> float:
    patterns = [
        r'\$?([\d,]+(?:\.\d+)?)',
        r'([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)',
        r'([\d,]+(?:\.\d+)?)%'
    ]
    # Apply patterns with validation
```

### Pattern 3: Sequence Labeling
```python
# For NER/BIO tagging tasks
def extract_bio_sequence(self, text: str) -> List[str]:
    # Extract and validate BIO sequences
    # Ensure I- tags follow B- tags
    # Handle entity boundaries
```

### Pattern 4: Multi-Attribute
```python
# For tasks with multiple output attributes
def extract_attributes(self, text: str) -> Dict[str, Any]:
    attributes = {}
    for attr in self.attributes:
        attributes[attr] = self.extract_single_attribute(text, attr)
    return attributes
```

## Conclusion

This migration plan provides a systematic approach to transitioning all FLAME tasks to BenchForge while:
- Preserving FLAME's financial domain expertise
- Enhancing extraction robustness (>95% success rate)
- Maintaining backward compatibility
- Enabling gradual, risk-free migration
- Setting foundation for future extensibility

The phased approach ensures minimal disruption while delivering significant improvements in reliability, performance, and maintainability.

**Estimated Total Timeline**: 8 weeks for complete migration
**Current Progress**: 50% (12/24 tasks migrated or code-ready)
**Next Immediate Step**: Complete testing of code-ready tasks and begin Phase 3