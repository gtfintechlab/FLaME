# BenchForge Documentation

Welcome to the BenchForge documentation! BenchForge is a professional benchmark engine for language models with first-class support for FLAME (Financial Language Model Evaluation).

## 📚 Documentation Index

### Getting Started
- **[Quick Start Guide](./QUICK_START_GUIDE.md)** - Get up and running quickly
- **[Installation](../README.md#installation)** - Installation instructions
- **[Examples](../examples/)** - Code examples and use cases

### Core Documentation
- **[Architecture](./ARCHITECTURE.md)** - System design and components
- **[API Reference](./API_REFERENCE.md)** - Complete API documentation
- **[Integration Guide](./INTEGRATION_GUIDE.md)** - General integration patterns and migration
- **[FLAME Guide](./FLAME_GUIDE.md)** - FLAME-specific features and examples

### Technical Specifications
- **[Master Plan](./BENCHFORGE_MASTER_PLAN.md)** - Overall project vision and roadmap
- **[Output Organization](./output_organization_spec.md)** - Output format specifications
- **[API Specification](./api_specification.md)** - Detailed API specifications
- **[Usage as Library](./usage_as_library.md)** - Using BenchForge as a Python library

### Development History
The following documents detail the development phases and are preserved for historical reference:

#### Phase 1: Core Engine
- [Phase 1 Core Engine](./phase1_core_engine.md) - Initial core engine design
- [Phase 1 Complete](./PHASE1_COMPLETE.md) - Phase 1 completion summary
- [Phase 1 Professional](./PHASE1_PROFESSIONAL_COMPLETE.md) - Professional enhancements

#### Phase 2: Support Systems
- [Phase 2 Support Systems](./phase2_support_systems.md) - Prompts and metrics

#### Phase 3: Data & Configuration
- [Phase 3 Data Config](./phase3_data_config.md) - Data management design
- [Phase 3 Completion](./phase3_completion_summary.md) - Phase 3 summary

#### Phase 4: FLAME Integration
- [Phase 4 FLAME Integration](./phase4_flame_integration.md) - FLAME integration design
- [Phase 4 Completion](./phase4_completion_summary.md) - Phase 4 summary

## 🎯 Quick Navigation

### By Use Case

#### "I want to run a benchmark"
→ Start with the [Quick Start Guide](./QUICK_START_GUIDE.md)

#### "I want to integrate my benchmark with BenchForge"
→ See [Integration Guide](./INTEGRATION_GUIDE.md)

#### "I want to integrate FLAME tasks"
→ See [FLAME Guide](./FLAME_GUIDE.md)

#### "I want to understand the architecture"
→ Read [Architecture Documentation](./ARCHITECTURE.md)

#### "I need API details"
→ Check the [API Reference](./API_REFERENCE.md)

#### "I want to contribute"
→ Review [Architecture](./ARCHITECTURE.md) and [Master Plan](./BENCHFORGE_MASTER_PLAN.md)

### By Component

| Component | Documentation |
|-----------|--------------|
| **Tasks** | [API Reference - Task System](./API_REFERENCE.md#task-system) |
| **LLM Interface** | [API Reference - LLM Interface](./API_REFERENCE.md#llm-interface) |
| **Prompts** | [API Reference - Prompt Management](./API_REFERENCE.md#prompt-management) |
| **Metrics** | [API Reference - Metrics System](./API_REFERENCE.md#metrics-system) |
| **Data** | [API Reference - Data Management](./API_REFERENCE.md#data-management) |
| **Integration** | [Integration Guide](./INTEGRATION_GUIDE.md) |
| **FLAME** | [FLAME Guide](./FLAME_GUIDE.md) |

## 📖 Documentation Standards

### Code Examples
All code examples in our documentation:
- Are tested and working
- Include necessary imports
- Show both basic and advanced usage
- Include error handling where appropriate

### API Documentation
Each API entry includes:
- Clear description of purpose
- Complete parameter documentation
- Return type specification
- Usage examples
- Related methods/classes

## 🔧 Building Documentation

To build a local copy of the documentation:

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material

# Serve documentation locally
mkdocs serve

# Build static documentation
mkdocs build
```

## 📝 Contributing to Documentation

We welcome documentation improvements! Please:
1. Follow the existing documentation style
2. Include code examples where helpful
3. Update the index when adding new documents
4. Verify all links work correctly
5. Test code examples before submitting

## 🆘 Getting Help

- **Issues**: [GitHub Issues](https://github.com/benchforge/benchforge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/benchforge/benchforge/discussions)
- **Examples**: [Example Code](../examples/)

## 📜 License

The documentation is provided under the same [MIT License](../LICENSE.txt) as the BenchForge project.

---

*Last updated: August 2024*
*BenchForge Version: 0.4.0*