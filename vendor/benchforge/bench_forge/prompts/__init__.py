"""Prompt management system for BenchForge."""

from bench_forge.prompts.templates import (
    PromptTemplate,
    ZeroShotTemplate,
    FewShotTemplate,
    ChainOfThoughtTemplate,
    InstructionTemplate,
    CustomTemplate,
    create_template,
)
from bench_forge.prompts.formats import (
    PromptFormat,
    ResponseFormat,
    ExtractionStrategy,
    PromptComponents,
    PromptExample,
)
from bench_forge.prompts.registry import (
    PromptRegistry,
    get_prompt_registry,
    register_prompt,
    prompt,
)
from bench_forge.prompts.extractor import (
    ResponseExtractor,
    ExtractionResult,
    get_extractor,
)

__all__ = [
    # Templates
    "PromptTemplate",
    "ZeroShotTemplate",
    "FewShotTemplate",
    "ChainOfThoughtTemplate",
    "InstructionTemplate",
    "CustomTemplate",
    "create_template",
    # Formats
    "PromptFormat",
    "ResponseFormat",
    "ExtractionStrategy",
    "PromptComponents",
    "PromptExample",
    # Registry
    "PromptRegistry",
    "get_prompt_registry",
    "register_prompt",
    "prompt",
    # Extraction
    "ResponseExtractor",
    "ExtractionResult",
    "get_extractor",
]
