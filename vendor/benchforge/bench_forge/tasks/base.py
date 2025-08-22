"""Base task interface with proper abstractions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

from bench_forge.tasks.config import TaskConfig, PromptFormat


logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """Abstract base class for benchmark tasks.

    All benchmark tasks must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, config: Optional[TaskConfig] = None):
        """Initialize task with configuration.

        Args:
            config: Task configuration. If None, must be set before use.
        """
        self.config = config
        self._dataset = None
        self._results = None

        if config:
            logger.info(f"Initialized task: {config.name}")

    @property
    def name(self) -> str:
        """Get task name from config."""
        if not self.config:
            raise RuntimeError("Task not configured")
        return self.config.name

    @abstractmethod
    def load_dataset(self, split: Optional[str] = None) -> Any:
        """Load dataset for this task.

        Args:
            split: Dataset split to load. If None, uses config.dataset_split

        Returns:
            Loaded dataset (format depends on implementation)
        """
        pass

    @abstractmethod
    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create a prompt for a single sample.

        Args:
            sample: Single dataset sample
            format: Prompt format to use. If None, uses config.prompt_format

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract structured answer from model response.

        Args:
            raw_response: Raw model output
            sample: Original sample (for context if needed)

        Returns:
            Extracted answer in appropriate format
        """
        pass

    def prepare_prompts(
        self, dataset: Any, format: Optional[PromptFormat] = None
    ) -> List[Dict[str, Any]]:
        """Prepare all prompts from dataset.

        Args:
            dataset: Loaded dataset
            format: Prompt format to use

        Returns:
            List of prompt dictionaries with 'prompt' and 'metadata' keys
        """
        if not self.config:
            raise RuntimeError("Task not configured")

        format = format or self.config.prompt_format
        prompts = []

        # Convert dataset to list if needed
        if not isinstance(dataset, list):
            dataset = list(dataset)

        # Limit samples if configured
        if self.config.num_samples:
            dataset = dataset[: self.config.num_samples]

        for idx, sample in enumerate(dataset):
            try:
                prompt_text = self.create_prompt(sample, format)
                prompts.append(
                    {
                        "prompt": prompt_text,
                        "metadata": {
                            "index": idx,
                            "sample": sample,
                            "format": format.value
                            if isinstance(format, PromptFormat)
                            else format,
                        },
                    }
                )
            except Exception as e:
                logger.error(f"Failed to create prompt for sample {idx}: {e}")
                # Add empty prompt to maintain alignment
                prompts.append(
                    {
                        "prompt": "",
                        "metadata": {
                            "index": idx,
                            "sample": sample,
                            "format": format.value
                            if isinstance(format, PromptFormat)
                            else format,
                            "error": str(e),
                        },
                    }
                )

        logger.info(f"Prepared {len(prompts)} prompts")
        return prompts

    def process_responses(
        self, responses: List[str], prompts: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Process model responses into structured results.

        Args:
            responses: List of model responses
            prompts: Original prompts with metadata

        Returns:
            DataFrame with processed results
        """
        results = []

        for response, prompt_data in zip(responses, prompts):
            sample = prompt_data["metadata"]["sample"]

            # Handle extraction errors gracefully
            try:
                extracted = self.extract_response(response, sample)
            except Exception as e:
                logger.error(
                    f"Failed to extract response for sample {prompt_data['metadata']['index']}: {e}"
                )
                extracted = None

            result = {
                "index": prompt_data["metadata"]["index"],
                "input": sample,
                "prompt": prompt_data["prompt"],
                "raw_response": response,
                "extracted_response": extracted,
                "format": prompt_data["metadata"]["format"],
                "ground_truth": sample.get("label", sample.get("answer", None)),
            }

            # Add any error information
            if "error" in prompt_data["metadata"]:
                result["prompt_error"] = prompt_data["metadata"]["error"]

            results.append(result)

        df = pd.DataFrame(results)
        logger.info(f"Processed {len(df)} responses")
        return df

    def validate_config(self) -> bool:
        """Validate task configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config:
            raise ValueError("Task configuration not set")

        if not self.config.name:
            raise ValueError("Task name is required")

        return True

    def get_metrics(self) -> List[str]:
        """Get list of metrics for this task."""
        if not self.config:
            return []
        return self.config.metrics or []
