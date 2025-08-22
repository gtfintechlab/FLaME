"""Prompt template system with multiple formats and validation."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
from string import Template
import logging

from bench_forge.prompts.formats import (
    PromptFormat,
    PromptComponents,
    PromptExample,
    ResponseFormat,
)


logger = logging.getLogger(__name__)


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    def __init__(
        self,
        components: Optional[PromptComponents] = None,
        format_type: PromptFormat = PromptFormat.ZERO_SHOT,
        response_format: ResponseFormat = ResponseFormat.TEXT,
    ):
        """Initialize template with components.

        Args:
            components: Prompt components
            format_type: Prompt format type
            response_format: Expected response format
        """
        self.components = components or PromptComponents()
        self.format_type = format_type
        self.response_format = response_format
        self._template_cache = {}

    @abstractmethod
    def render(self, **kwargs) -> str:
        """Render the prompt to a string.

        Args:
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string
        """
        pass

    def validate(self) -> bool:
        """Validate template configuration.

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        return self.components.validate()

    def _format_examples(
        self,
        input_label: str = "Input",
        output_label: str = "Output",
        separator: str = "\n\n",
    ) -> str:
        """Format examples for few-shot prompting.

        Args:
            input_label: Label for inputs
            output_label: Label for outputs
            separator: Separator between examples

        Returns:
            Formatted examples string
        """
        if not self.components.examples:
            return ""

        formatted_examples = []
        for i, example in enumerate(self.components.examples, 1):
            example_str = example.format(input_label, output_label)
            formatted_examples.append(example_str)

        return separator.join(formatted_examples)

    def _format_constraints(self, bullet: str = "- ") -> str:
        """Format constraints as a list.

        Args:
            bullet: Bullet point style

        Returns:
            Formatted constraints
        """
        if not self.components.constraints:
            return ""

        return "\n".join([f"{bullet}{c}" for c in self.components.constraints])


class ZeroShotTemplate(PromptTemplate):
    """Zero-shot prompt template."""

    def render(self, **kwargs) -> str:
        """Render zero-shot prompt.

        Args:
            **kwargs: Template variables

        Returns:
            Formatted prompt
        """
        parts = []

        # System message
        if self.components.system_message:
            parts.append(f"System: {self.components.system_message}")

        # Instruction
        if self.components.instruction:
            parts.append(self.components.instruction)

        # Context
        if self.components.context:
            parts.append(f"Context: {self.components.context}")

        # Constraints
        if self.components.constraints:
            parts.append("Requirements:")
            parts.append(self._format_constraints())

        # Output format
        if self.components.output_format:
            parts.append(f"Output format: {self.components.output_format}")

        # Input
        if self.components.input_text:
            parts.append(f"Input: {self.components.input_text}")

        # Add custom kwargs
        for key, value in kwargs.items():
            if value is not None:
                parts.append(f"{key}: {value}")

        prompt = "\n\n".join(filter(None, parts))

        logger.debug(f"Rendered zero-shot prompt: {len(prompt)} chars")
        return prompt


class FewShotTemplate(PromptTemplate):
    """Few-shot prompt template with examples."""

    def __init__(
        self,
        components: Optional[PromptComponents] = None,
        num_examples: Optional[int] = None,
        example_selection: str = "first",  # first, last, random, diverse
        **kwargs,
    ):
        """Initialize few-shot template.

        Args:
            components: Prompt components
            num_examples: Max number of examples to use
            example_selection: How to select examples
            **kwargs: Additional arguments
        """
        super().__init__(components, PromptFormat.FEW_SHOT, **kwargs)
        self.num_examples = num_examples
        self.example_selection = example_selection

    def render(self, **kwargs) -> str:
        """Render few-shot prompt with examples.

        Args:
            **kwargs: Template variables

        Returns:
            Formatted prompt
        """
        parts = []

        # System message
        if self.components.system_message:
            parts.append(f"System: {self.components.system_message}")

        # Instruction
        if self.components.instruction:
            parts.append(self.components.instruction)

        # Context
        if self.components.context:
            parts.append(f"Context: {self.components.context}")

        # Examples
        if self.components.examples:
            parts.append("Examples:")
            self._select_examples()
            parts.append(self._format_examples())

        # Constraints
        if self.components.constraints:
            parts.append("Requirements:")
            parts.append(self._format_constraints())

        # Output format
        if self.components.output_format:
            parts.append(f"Output format: {self.components.output_format}")

        # Now the actual input
        parts.append("Now process the following:")

        # Input
        if self.components.input_text:
            parts.append(f"Input: {self.components.input_text}")

        # Add custom kwargs
        for key, value in kwargs.items():
            if value is not None:
                parts.append(f"{key}: {value}")

        prompt = "\n\n".join(filter(None, parts))

        logger.debug(
            f"Rendered few-shot prompt: {len(prompt)} chars, {len(self.components.examples)} examples"
        )
        return prompt

    def _select_examples(self) -> List[PromptExample]:
        """Select examples based on strategy.

        Returns:
            Selected examples
        """
        examples = self.components.examples

        if not self.num_examples or self.num_examples >= len(examples):
            return examples

        if self.example_selection == "first":
            return examples[: self.num_examples]
        elif self.example_selection == "last":
            return examples[-self.num_examples :]
        elif self.example_selection == "random":
            import random

            return random.sample(examples, self.num_examples)
        else:
            # Default to first
            return examples[: self.num_examples]


class ChainOfThoughtTemplate(PromptTemplate):
    """Chain-of-thought prompt template."""

    def __init__(
        self,
        components: Optional[PromptComponents] = None,
        reasoning_steps: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize CoT template.

        Args:
            components: Prompt components
            reasoning_steps: Optional reasoning steps to include
            **kwargs: Additional arguments
        """
        super().__init__(components, PromptFormat.CHAIN_OF_THOUGHT, **kwargs)
        self.reasoning_steps = reasoning_steps or []

    def render(self, **kwargs) -> str:
        """Render chain-of-thought prompt.

        Args:
            **kwargs: Template variables

        Returns:
            Formatted prompt
        """
        parts = []

        # System message
        if self.components.system_message:
            parts.append(f"System: {self.components.system_message}")

        # Instruction with CoT emphasis
        if self.components.instruction:
            parts.append(self.components.instruction)
            parts.append("Let's think step by step.")

        # Context
        if self.components.context:
            parts.append(f"Context: {self.components.context}")

        # Examples with reasoning
        if self.components.examples:
            parts.append("Examples with reasoning:")
            for i, example in enumerate(self.components.examples, 1):
                parts.append(f"Example {i}:")
                parts.append(f"Input: {example.input_text}")
                if example.explanation:
                    parts.append(f"Reasoning: {example.explanation}")
                parts.append(f"Output: {example.output_text}")
                parts.append("")

        # Reasoning steps if provided
        if self.reasoning_steps:
            parts.append("Follow these reasoning steps:")
            for i, step in enumerate(self.reasoning_steps, 1):
                parts.append(f"{i}. {step}")

        # Constraints
        if self.components.constraints:
            parts.append("Requirements:")
            parts.append(self._format_constraints())

        # Output format
        if self.components.output_format:
            parts.append(f"Output format: {self.components.output_format}")

        # Input
        if self.components.input_text:
            parts.append(f"Input: {self.components.input_text}")
            parts.append("Let's work through this step by step:")

        # Add custom kwargs
        for key, value in kwargs.items():
            if value is not None:
                parts.append(f"{key}: {value}")

        prompt = "\n\n".join(filter(None, parts))

        logger.debug(f"Rendered chain-of-thought prompt: {len(prompt)} chars")
        return prompt


class InstructionTemplate(PromptTemplate):
    """Instruction-following prompt template."""

    def render(self, **kwargs) -> str:
        """Render instruction prompt.

        Args:
            **kwargs: Template variables

        Returns:
            Formatted prompt
        """
        parts = []

        # System message
        if self.components.system_message:
            parts.append(f"### System\n{self.components.system_message}")

        # Instruction
        if self.components.instruction:
            parts.append(f"### Instruction\n{self.components.instruction}")

        # Context
        if self.components.context:
            parts.append(f"### Context\n{self.components.context}")

        # Examples
        if self.components.examples:
            parts.append("### Examples")
            for i, example in enumerate(self.components.examples, 1):
                parts.append(f"Example {i}:")
                parts.append(f"Input: {example.input_text}")
                parts.append(f"Output: {example.output_text}")

        # Constraints
        if self.components.constraints:
            parts.append("### Requirements")
            parts.append(self._format_constraints())

        # Input
        if self.components.input_text:
            parts.append(f"### Input\n{self.components.input_text}")

        # Output instruction
        parts.append("### Response")

        prompt = "\n\n".join(filter(None, parts))

        logger.debug(f"Rendered instruction prompt: {len(prompt)} chars")
        return prompt


class CustomTemplate(PromptTemplate):
    """Custom template with user-defined format."""

    def __init__(
        self,
        template_string: str,
        components: Optional[PromptComponents] = None,
        safe_substitution: bool = True,
        **kwargs,
    ):
        """Initialize custom template.

        Args:
            template_string: Template string with placeholders
            components: Prompt components
            safe_substitution: Use safe substitution (ignores missing keys)
            **kwargs: Additional arguments
        """
        super().__init__(components, PromptFormat.CUSTOM, **kwargs)
        self.template_string = template_string
        self.safe_substitution = safe_substitution

    def render(self, **kwargs) -> str:
        """Render custom template.

        Args:
            **kwargs: Template variables

        Returns:
            Formatted prompt
        """
        # Prepare template variables
        template_vars = {
            "instruction": self.components.instruction or "",
            "context": self.components.context or "",
            "input": self.components.input_text or "",
            "output_format": self.components.output_format or "",
            "constraints": self._format_constraints(),
            "examples": self._format_examples(),
            "system": self.components.system_message or "",
        }

        # Add component metadata
        template_vars.update(self.components.metadata)

        # Add custom kwargs
        template_vars.update(kwargs)

        # Render template
        if self.safe_substitution:
            template = Template(self.template_string)
            prompt = template.safe_substitute(**template_vars)
        else:
            prompt = self.template_string.format(**template_vars)

        logger.debug(f"Rendered custom prompt: {len(prompt)} chars")
        return prompt


# Factory function
def create_template(
    format_type: Union[str, PromptFormat],
    components: Optional[PromptComponents] = None,
    **kwargs,
) -> PromptTemplate:
    """Create a prompt template of the specified type.

    Args:
        format_type: Type of template to create
        components: Prompt components
        **kwargs: Additional template-specific arguments

    Returns:
        PromptTemplate instance

    Raises:
        ValueError: If format_type is invalid
    """
    if isinstance(format_type, str):
        try:
            format_type = PromptFormat(format_type)
        except ValueError:
            raise ValueError(f"Invalid format type: {format_type}")

    if format_type == PromptFormat.ZERO_SHOT:
        return ZeroShotTemplate(components, **kwargs)
    elif format_type == PromptFormat.FEW_SHOT:
        return FewShotTemplate(components, **kwargs)
    elif format_type == PromptFormat.CHAIN_OF_THOUGHT:
        return ChainOfThoughtTemplate(components, **kwargs)
    elif format_type == PromptFormat.INSTRUCTION:
        return InstructionTemplate(components, **kwargs)
    elif format_type == PromptFormat.CUSTOM:
        if "template_string" not in kwargs:
            raise ValueError("Custom template requires template_string")
        return CustomTemplate(components=components, **kwargs)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
