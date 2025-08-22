"""Prompt format definitions and utilities."""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging


logger = logging.getLogger(__name__)


class PromptFormat(Enum):
    """Supported prompt formats."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    INSTRUCTION = "instruction"
    SYSTEM = "system"
    CUSTOM = "custom"


class ResponseFormat(Enum):
    """Expected response formats."""

    TEXT = "text"
    JSON = "json"
    LABEL = "label"
    NUMBER = "number"
    LIST = "list"
    STRUCTURED = "structured"
    BOOLEAN = "boolean"


class ExtractionStrategy(Enum):
    """Response extraction strategies."""

    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    REGEX = "regex"
    FIRST_LINE = "first_line"
    LAST_LINE = "last_line"
    JSON_PARSE = "json_parse"
    KEYWORD = "keyword"
    FUZZY_MATCH = "fuzzy_match"


@dataclass
class PromptExample:
    """A single example for few-shot prompting."""

    input_text: str
    output_text: str
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"input": self.input_text, "output": self.output_text}
        if self.explanation:
            result["explanation"] = self.explanation
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def format(self, input_label: str = "Input", output_label: str = "Output") -> str:
        """Format example as string.

        Args:
            input_label: Label for input
            output_label: Label for output

        Returns:
            Formatted example string
        """
        parts = [f"{input_label}: {self.input_text}"]

        if self.explanation:
            parts.append(f"Reasoning: {self.explanation}")

        parts.append(f"{output_label}: {self.output_text}")

        return "\n".join(parts)


@dataclass
class PromptComponents:
    """Standard components of a prompt with validation."""

    instruction: Optional[str] = None
    context: Optional[str] = None
    examples: List[PromptExample] = field(default_factory=list)
    input_text: Optional[str] = None
    output_format: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    system_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate components."""
        # Convert dict examples to PromptExample objects
        converted_examples = []
        for example in self.examples:
            if isinstance(example, dict):
                try:
                    converted_examples.append(
                        PromptExample(
                            input_text=example.get(
                                "input", example.get("input_text", "")
                            ),
                            output_text=example.get(
                                "output", example.get("output_text", "")
                            ),
                            explanation=example.get("explanation"),
                            metadata=example.get("metadata", {}),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to convert example: {e}")
            elif isinstance(example, PromptExample):
                converted_examples.append(example)
            else:
                logger.warning(f"Invalid example type: {type(example)}")

        self.examples = converted_examples

    def add_example(
        self,
        input_text: str,
        output_text: str,
        explanation: Optional[str] = None,
        **metadata,
    ):
        """Add a few-shot example.

        Args:
            input_text: Example input
            output_text: Example output
            explanation: Optional explanation
            **metadata: Additional metadata
        """
        example = PromptExample(
            input_text=input_text,
            output_text=output_text,
            explanation=explanation,
            metadata=metadata,
        )
        self.examples.append(example)
        logger.debug(f"Added example: {len(self.examples)} total")

    def add_constraint(self, constraint: str):
        """Add a constraint or rule.

        Args:
            constraint: Constraint text
        """
        if constraint and constraint not in self.constraints:
            self.constraints.append(constraint)
            logger.debug(f"Added constraint: {constraint}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instruction": self.instruction,
            "context": self.context,
            "examples": [ex.to_dict() for ex in self.examples],
            "input_text": self.input_text,
            "output_format": self.output_format,
            "constraints": self.constraints,
            "system_message": self.system_message,
            "metadata": self.metadata,
        }

    def validate(self) -> bool:
        """Validate components.

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        # At least one component should be present
        if not any(
            [
                self.instruction,
                self.context,
                self.examples,
                self.input_text,
                self.system_message,
            ]
        ):
            raise ValueError("Prompt must have at least one component")

        # Validate examples
        for i, example in enumerate(self.examples):
            if not example.input_text or not example.output_text:
                raise ValueError(f"Example {i} missing input or output")

        return True
