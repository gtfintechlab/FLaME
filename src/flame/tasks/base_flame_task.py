"""Base class for FLAME tasks using BenchForge.

This module provides the foundation for all FLAME tasks,
leveraging BenchForge's professional infrastructure.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from flame.benchforge import (
    FLAMETask,
    FLAMEConfig,
    PromptFormat,
    BENCHFORGE_AVAILABLE,
)

logger = logging.getLogger(__name__)


class BaseFLAMETask(FLAMETask):
    """Base class for FLAME financial benchmark tasks.

    This class extends BenchForge's FLAMETask with additional
    functionality specific to financial language understanding evaluation.
    """

    def __init__(self, config: Optional[FLAMEConfig] = None):
        """Initialize FLAME task.

        Args:
            config: Task configuration
        """
        if not BENCHFORGE_AVAILABLE:
            raise ImportError(
                "BenchForge is required for FLAME tasks. "
                "Install with: pip install -e ./benchforge"
            )

        super().__init__(config)

        # FLAME-specific attributes
        self.financial_domain = getattr(config, "financial_domain", "general")
        self.dataset_loaded = False

        logger.info(
            f"Initialized FLAME task: {self.config.name} (domain: {self.financial_domain})"
        )

    @abstractmethod
    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for sample.

        Must be implemented by specific FLAME tasks.

        Args:
            sample: Dataset sample
            format: Prompt format

        Returns:
            Formatted prompt
        """
        pass

    def create_zero_shot_prompt(self, sample: Dict[str, Any]) -> str:
        """Create zero-shot prompt.

        Default implementation that can be overridden.

        Args:
            sample: Dataset sample

        Returns:
            Zero-shot prompt
        """
        text = sample.get(self.config.text_field, "")

        if self.config.valid_labels:
            labels_str = "/".join(self.config.valid_labels)
            return f"""Classify the following text as {labels_str}.

Text: {text}

Classification:"""
        else:
            return f"""Analyze the following text:

Text: {text}

Analysis:"""

    def create_few_shot_prompt(
        self, sample: Dict[str, Any], examples: List[Dict[str, Any]] = None
    ) -> str:
        """Create few-shot prompt.

        Default implementation that can be overridden.

        Args:
            sample: Dataset sample
            examples: Optional examples for few-shot learning

        Returns:
            Few-shot prompt
        """
        if examples is None:
            # Use default examples if available
            examples = self.get_default_examples()

        prompt_parts = ["Here are some examples:"]

        for i, example in enumerate(examples, 1):
            text = example.get(self.config.text_field, "")
            label = example.get(self.config.label_field, "")
            prompt_parts.append(f"""
Example {i}:
Text: {text}
Answer: {label}""")

        # Add the actual sample
        text = sample.get(self.config.text_field, "")
        prompt_parts.append(f"""
Now classify:
Text: {text}
Answer:""")

        return "\n".join(prompt_parts)

    def create_chain_of_thought_prompt(self, sample: Dict[str, Any]) -> str:
        """Create chain-of-thought prompt.

        Default implementation that can be overridden.

        Args:
            sample: Dataset sample

        Returns:
            Chain-of-thought prompt
        """
        text = sample.get(self.config.text_field, "")

        if self.config.valid_labels:
            labels_str = "/".join(self.config.valid_labels)
            return f"""Classify the following text as {labels_str}.

Think step-by-step about the classification:
1. Identify key phrases and their implications
2. Consider the overall context
3. Determine the most appropriate classification

Text: {text}

Let's think step by step:"""
        else:
            return f"""Analyze the following text step-by-step:

Text: {text}

Step-by-step analysis:"""

    def get_default_examples(self) -> List[Dict[str, Any]]:
        """Get default examples for few-shot prompting.

        Can be overridden by specific tasks.

        Returns:
            List of example samples
        """
        return []

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a dataset sample.

        Args:
            sample: Sample to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if self.config.text_field not in sample:
            logger.warning(f"Sample missing text field: {self.config.text_field}")
            return False

        # Check if text is non-empty
        text = sample[self.config.text_field]
        if not text or (isinstance(text, str) and not text.strip()):
            logger.warning("Sample has empty text")
            return False

        return True

    def preprocess_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a sample before prompt creation.

        Can be overridden for task-specific preprocessing.

        Args:
            sample: Original sample

        Returns:
            Preprocessed sample
        """
        # Default: just ensure text is string
        processed = sample.copy()

        if self.config.text_field in processed:
            text = processed[self.config.text_field]
            if not isinstance(text, str):
                processed[self.config.text_field] = str(text)

        return processed

    def postprocess_response(self, response: str) -> str:
        """Postprocess model response.

        Can be overridden for task-specific postprocessing.

        Args:
            response: Raw model response

        Returns:
            Postprocessed response
        """
        # Default: just strip whitespace
        return response.strip() if response else ""

    def compute_task_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Compute task-specific metrics.

        Can be overridden for custom metrics.

        Args:
            results_df: Results DataFrame

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic accuracy if ground truth available
        if (
            "extracted_response" in results_df.columns
            and "ground_truth" in results_df.columns
        ):
            valid_mask = (
                results_df["extracted_response"].notna()
                & results_df["ground_truth"].notna()
            )
            valid_df = results_df[valid_mask]

            if len(valid_df) > 0:
                correct = (
                    valid_df["extracted_response"] == valid_df["ground_truth"]
                ).sum()
                metrics["accuracy"] = correct / len(valid_df)
                metrics["error_rate"] = 1 - metrics["accuracy"]

        # Extraction success rate
        if "extracted_response" in results_df.columns:
            extraction_success = results_df["extracted_response"].notna().sum()
            metrics["extraction_rate"] = extraction_success / len(results_df)

        return metrics

    def get_task_info(self) -> Dict[str, Any]:
        """Get task information and metadata.

        Returns:
            Dictionary of task information
        """
        info = {
            "name": self.config.name,
            "dataset": self.config.dataset,
            "huggingface_dataset": self.config.huggingface_dataset,
            "financial_domain": self.financial_domain,
            "valid_labels": self.config.valid_labels,
            "text_field": self.config.text_field,
            "label_field": self.config.label_field,
            "extraction_strategy": self.config.extraction_strategy.value,
            "prompt_format": self.config.prompt_format.value,
            "batch_size": self.config.batch_size,
            "max_tokens": self.config.max_tokens,
        }

        # Add statistics if available
        stats = self.get_stats()
        if stats:
            info["statistics"] = stats

        return info
