# Changelog

All notable changes to FLAME (Financial Language Analysis and Modeling Engine) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - In Development

### Overview
This version focuses on migrating FLAME to use BenchForge infrastructure for improved reliability and performance.

### Work in Progress
- **BenchForge Integration**: Adding BenchForge as a git submodule for enhanced task execution
- **Task Migration**: Migrating FLAME tasks to BenchForge architecture
  - FOMC task successfully tested with improved extraction
  - Migration plan documented in `/docs/FLAME_TO_BENCHFORGE_MIGRATION_PLAN.md`
  - TODO tracking in `/docs/FLAME_MIGRATION_TODO.md`
- **Documentation**: Creating migration guides and implementation patterns

### Target Improvements
- Enhanced extraction reliability through multi-strategy approach
- Better parallelization and batch processing
- Gradual migration path with feature flags

### Note
This is a development version. Features and improvements are subject to change as migration progresses.

## [1.0.0] - 2024-12-01 (Main Branch Release)

### Added
- **Complete FLAME Framework**: Full implementation of Financial Language Analysis and Modeling Engine
- **24 Financial NLP Tasks**: Comprehensive task suite including:
  - Sentiment Analysis (FOMC, FPB, TSA, FiQA-SA)
  - Question Answering (ConvFinQA, FinQA, EDTSum, TATQA, FinQABench)
  - Named Entity Recognition (FiNER, FinEntity, NER)
  - Classification (Headlines, CD, SC, NCC, MA, MLESG, FLS)
  - Benchmark Suites (CFA, FINEVAL, Flare variants)
- **Multi-Provider Support**: Integration with OpenAI, Anthropic, Together AI, Ollama via LiteLLM
- **HuggingFace Integration**: All datasets hosted on gtfintechlab HuggingFace repository
- **Unified CLI**: Single entry point for all tasks via main.py
- **Batch Processing**: Efficient batch inference and evaluation
- **Comprehensive Evaluation**: Task-specific metrics and evaluation framework
- **Local Development**: Ollama support for cost-effective local testing

### Technical Foundation
- **Python 3.11+** requirement
- **Modular Architecture**: Clean separation of tasks, inference, and evaluation
- **Extensible Design**: Easy addition of new tasks and models
- **Robust Error Handling**: Retry logic and graceful failure modes
- **Performance Optimization**: Batch processing and caching strategies

## Migration Notes

### Upgrading from 1.0.0 to 1.1.0
1. Run `git submodule update --init --recursive` to fetch BenchForge
2. Set environment variables for task routing (e.g., `USE_BENCHFORGE_FOMC=1`)
3. Review migration documentation in `/docs/` for task-specific changes
4. Test with A/B comparison mode before full cutover

### Breaking Changes
- None - full backward compatibility maintained through compatibility layer

### Known Issues
- Token alignment for complex NER tasks requires additional testing
- Some benchmark tasks (CFA, FINEVAL) not yet migrated

### Roadmap for 1.2.0
- Complete migration of remaining 12 tasks
- Achieve 90% test coverage
- Implement automatic rollback triggers
- Add performance monitoring dashboard