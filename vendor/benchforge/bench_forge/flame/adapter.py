"""Professional FLAME adapter for BenchForge integration.

This module provides first-class FLAME support in BenchForge,
enabling seamless financial benchmark evaluation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from abc import abstractmethod

import pandas as pd

from bench_forge.tasks.base import BaseTask
from bench_forge.tasks.config import TaskConfig, PromptFormat
from bench_forge.tasks.registry import get_registry
from bench_forge.prompts.extractor import ResponseExtractor, ExtractionStrategy
from bench_forge.llm.config import LLMConfig
from bench_forge.data.loader import HuggingFaceLoader, LoaderConfig
from bench_forge.utils.validation import InputValidator

logger = logging.getLogger(__name__)


@dataclass
class FLAMEConfig(TaskConfig):
    """Extended configuration for FLAME tasks.

    Adds FLAME-specific configuration options while maintaining
    compatibility with base TaskConfig.
    """

    # FLAME-specific fields
    huggingface_dataset: Optional[str] = None
    label_field: str = "label"
    text_field: str = "text"
    valid_labels: List[str] = field(default_factory=list)
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.KEYWORD
    results_dir: Optional[Path] = None
    evaluation_dir: Optional[Path] = None

    # Financial task specific
    financial_domain: Optional[str] = (
        None  # e.g., "sentiment", "classification", "extraction"
    )
    regulatory_compliance: bool = False
    include_confidence: bool = False

    def __post_init__(self):
        """Post-initialization setup."""
        super().__post_init__()

        # Set default paths if not provided
        if self.results_dir is None:
            self.results_dir = Path("results") / self.name
        if self.evaluation_dir is None:
            self.evaluation_dir = Path("evaluations") / self.name

        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)


class FLAMETask(BaseTask):
    """Base class for FLAME tasks with BenchForge integration.

    This class provides FLAME-specific functionality while leveraging
    BenchForge's professional infrastructure.
    """

    def __init__(self, config: Optional[FLAMEConfig] = None):
        """Initialize FLAME task.

        Args:
            config: FLAME task configuration
        """
        # Ensure we have FLAMEConfig
        if config is None:
            config = FLAMEConfig(name="unknown_flame_task")
        elif not isinstance(config, FLAMEConfig):
            # Convert regular TaskConfig to FLAMEConfig
            flame_config = FLAMEConfig(**config.__dict__)
            config = flame_config

        super().__init__(config)

        # Initialize FLAME-specific components
        self.config: FLAMEConfig = config
        self.extractor = ResponseExtractor(default_strategy=config.extraction_strategy)
        self.validator = InputValidator()

        # Cache for dataset
        self._dataset_cache = {}

        # Statistics
        self._stats = {
            "prompts_created": 0,
            "responses_extracted": 0,
            "extraction_failures": 0,
        }

        logger.info(f"Initialized FLAME task: {config.name}")

    def load_dataset(self, split: Optional[str] = None) -> Any:
        """Load dataset with caching support.

        Args:
            split: Dataset split to load

        Returns:
            Loaded dataset
        """
        split = split or self.config.dataset_split
        cache_key = f"{self.config.huggingface_dataset}_{split}"

        # Check cache
        if cache_key in self._dataset_cache:
            logger.debug(f"Using cached dataset: {cache_key}")
            return self._dataset_cache[cache_key]

        # Load from HuggingFace
        if self.config.huggingface_dataset:
            loader_config = LoaderConfig(
                validate_on_load=True, cache_dir=Path(".cache") / "datasets"
            )
            loader = HuggingFaceLoader(loader_config)

            try:
                dataset = loader.load(
                    self.config.huggingface_dataset, split=split, trust_remote_code=True
                )

                # Validate dataset
                validation = self.validator.validate_dataset(
                    dataset,
                    required_fields=[self.config.text_field, self.config.label_field],
                )

                if not validation.is_valid:
                    logger.warning(
                        f"Dataset validation warnings: {validation.warnings}"
                    )

                # Cache dataset
                self._dataset_cache[cache_key] = dataset

                logger.info(
                    f"Loaded dataset: {self.config.huggingface_dataset} ({len(dataset)} samples)"
                )
                return dataset

            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise
        else:
            raise ValueError("No dataset specified in configuration")

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

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract structured response using configured strategy.

        Args:
            raw_response: Raw model output
            sample: Original sample for context

        Returns:
            Extracted response
        """
        self._stats["responses_extracted"] += 1

        try:
            # Use valid labels if available
            if self.config.valid_labels:
                extracted = self.extractor.extract_label(
                    raw_response,
                    self.config.valid_labels,
                    strategy=self.config.extraction_strategy,
                )
            else:
                # General extraction
                result = self.extractor.extract(
                    raw_response, strategy=self.config.extraction_strategy
                )
                extracted = result.value

            if extracted is None:
                self._stats["extraction_failures"] += 1
                logger.debug(f"Extraction failed for response: {raw_response[:100]}...")

            return extracted

        except Exception as e:
            self._stats["extraction_failures"] += 1
            logger.error(f"Extraction error: {e}")
            return None

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth from sample.

        Args:
            sample: Dataset sample

        Returns:
            Ground truth value
        """
        # Use configured label field
        if self.config.label_field in sample:
            return sample[self.config.label_field]

        # Fallback to common field names
        for field_name in ["label", "labels", "target", "answer", "ground_truth"]:
            if field_name in sample:
                return sample[field_name]

        logger.warning("No ground truth field found in sample")
        return None

    def process_batch(
        self, samples: List[Dict[str, Any]], format: Optional[PromptFormat] = None
    ) -> List[str]:
        """Process batch of samples into prompts.

        Args:
            samples: List of dataset samples
            format: Prompt format to use

        Returns:
            List of formatted prompts
        """
        prompts = []
        format = format or self.config.prompt_format

        for sample in samples:
            prompt = self.create_prompt(sample, format)
            prompts.append(prompt)
            self._stats["prompts_created"] += 1

        return prompts

    def format_results(
        self,
        samples: List[Dict[str, Any]],
        prompts: List[str],
        raw_responses: List[str],
        extracted_responses: List[Any],
    ) -> pd.DataFrame:
        """Format results into FLAME-compatible DataFrame.

        Args:
            samples: Original dataset samples
            prompts: Generated prompts
            raw_responses: Raw model outputs
            extracted_responses: Extracted structured responses

        Returns:
            Results DataFrame
        """
        # Build results DataFrame
        results = []

        for i, (sample, prompt, raw, extracted) in enumerate(
            zip(samples, prompts, raw_responses, extracted_responses)
        ):
            result = {
                "index": i,
                "input": sample.get(self.config.text_field, ""),
                "prompt": prompt,
                "raw_response": raw,
                "extracted_response": extracted,
                "ground_truth": self.get_ground_truth(sample),
            }

            # Add sample metadata
            for key, value in sample.items():
                if key not in [self.config.text_field, self.config.label_field]:
                    result[f"meta_{key}"] = value

            results.append(result)

        df = pd.DataFrame(results)

        # Add metadata to DataFrame
        df.attrs["task"] = self.config.name
        df.attrs["dataset"] = self.config.dataset
        df.attrs["model"] = getattr(self.config, "model", "unknown")
        df.attrs["prompt_format"] = self.config.prompt_format.value

        return df

    def get_stats(self) -> Dict[str, Any]:
        """Get task statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self._stats.copy()

        # Calculate rates
        if stats["responses_extracted"] > 0:
            stats["extraction_success_rate"] = 1 - (
                stats["extraction_failures"] / stats["responses_extracted"]
            )

        return stats


class FLAMEAdapter:
    """Main adapter for FLAME integration with BenchForge.

    This adapter provides high-level orchestration for FLAME workflows
    using BenchForge infrastructure.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize FLAME adapter.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.registry = get_registry()
        self._registered_tasks = {}

        logger.info("Initialized FLAME adapter")

    def register_task(
        self, name: str, task_class: type, config: Optional[FLAMEConfig] = None
    ):
        """Register a FLAME task with BenchForge.

        Args:
            name: Task name
            task_class: Task class (must inherit from FLAMETask)
            config: Optional task configuration
        """
        if not issubclass(task_class, FLAMETask):
            raise ValueError(f"Task class must inherit from FLAMETask: {task_class}")

        # Register with BenchForge registry
        self.registry.register(name, task_class)

        # Track registration
        self._registered_tasks[name] = {
            "class": task_class,
            "config": config,
        }

        logger.info(f"Registered FLAME task: {name}")

    def create_task(self, name: str, config: Optional[FLAMEConfig] = None) -> FLAMETask:
        """Create a FLAME task instance.

        Args:
            name: Task name
            config: Optional task configuration

        Returns:
            FLAMETask instance
        """
        # Get registered config if available
        if name in self._registered_tasks and config is None:
            config = self._registered_tasks[name]["config"]

        # Create task through registry
        task = self.registry.create_task(name, config)

        if not isinstance(task, FLAMETask):
            logger.warning(f"Task {name} is not a FLAMETask, wrapping it")
            # Could wrap non-FLAME tasks here if needed

        return task

    def list_tasks(self) -> List[str]:
        """List all registered FLAME tasks.

        Returns:
            List of task names
        """
        return list(self._registered_tasks.keys())

    def convert_args(self, args) -> FLAMEConfig:
        """Convert command-line arguments to FLAME configuration.

        Args:
            args: Argument namespace (e.g., from argparse)

        Returns:
            FLAMEConfig instance
        """
        # Extract task name
        task_name = getattr(args, "task", None) or getattr(args, "dataset", "unknown")

        # Determine prompt format
        prompt_format = PromptFormat.ZERO_SHOT
        if hasattr(args, "prompt_format"):
            format_str = args.prompt_format.lower()
            if "few" in format_str:
                prompt_format = PromptFormat.FEW_SHOT
            elif "chain" in format_str or "cot" in format_str:
                prompt_format = PromptFormat.CHAIN_OF_THOUGHT

        # Build configuration
        config = FLAMEConfig(
            name=task_name,
            dataset=getattr(args, "dataset", task_name),
            dataset_split=getattr(args, "split", "test"),
            metrics=getattr(args, "metrics", ["accuracy", "f1_macro"]),
            prompt_format=prompt_format,
            batch_size=getattr(args, "batch_size", 10),
            max_tokens=getattr(args, "max_tokens", 256),
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            top_k=getattr(args, "top_k", None),
            seed=getattr(args, "seed", 42),
            num_samples=getattr(args, "num_samples", None),
        )

        return config

    def process_batch_compatibility(
        self,
        args,
        messages_batch: List[Any],
        batch_idx: int,
        total_batches: int,
        llm_client=None,
    ) -> List[Any]:
        """Process batch with FLAME compatibility.

        This method provides backward compatibility for FLAME's
        existing batch processing interface.

        Args:
            args: Command-line arguments
            messages_batch: Batch of messages/prompts
            batch_idx: Current batch index
            total_batches: Total number of batches
            llm_client: Optional LLM client

        Returns:
            List of responses in FLAME format
        """
        from bench_forge.llm.client import LLMClient

        # Create LLM client if not provided
        if llm_client is None:
            llm_config = LLMConfig(
                provider="litellm",
                model=getattr(args, "model", "gpt-3.5-turbo"),
                max_tokens=getattr(args, "max_tokens", 256),
                temperature=getattr(args, "temperature", 0.0),
            )
            llm_client = LLMClient(llm_config)

        # Extract prompts from messages
        prompts = []
        for msg in messages_batch:
            if isinstance(msg, list) and len(msg) > 0:
                # Chat format - get last message content
                prompts.append(msg[-1].get("content", ""))
            elif isinstance(msg, str):
                prompts.append(msg)
            else:
                prompts.append(str(msg))

        logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")

        # Process through LLM
        responses = llm_client.complete_batch(prompts)

        # Convert to FLAME's expected format
        flame_responses = []
        for response in responses:
            # Create mock response object that FLAME expects
            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]

            class MockChoice:
                def __init__(self, content):
                    self.message = MockMessage(content)

            class MockMessage:
                def __init__(self, content):
                    self.content = content

            flame_responses.append(MockResponse(response))

        return flame_responses


def flame_task(name: str, config: Optional[FLAMEConfig] = None):
    """Decorator for registering FLAME tasks.

    Usage:
        @flame_task("my_task")
        class MyTask(FLAMETask):
            ...

    Args:
        name: Task name
        config: Optional default configuration

    Returns:
        Decorator function
    """

    def decorator(cls):
        # Validate class
        if not issubclass(cls, FLAMETask):
            raise ValueError(f"Class must inherit from FLAMETask: {cls}")

        # Register with global registry
        registry = get_registry()
        registry.register(name, cls)

        # Add metadata
        cls._flame_task_name = name
        cls._flame_default_config = config

        logger.info(f"Registered FLAME task via decorator: {name}")

        return cls

    return decorator
