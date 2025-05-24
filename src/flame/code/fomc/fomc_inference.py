"""FOMC inference module."""

import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from flame.utils.dataset_utils import safe_load_dataset
from pathlib import Path
from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import get_component_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.miscellaneous import generate_inference_filename


# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "fomc")


@dataclass
class InferenceConfig:
    """Configuration for FOMC inference."""

    model: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int]
    repetition_penalty: float
    batch_size: int

    def __post_init__(self):
        """Validate configuration values."""
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("Top_p must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")


def validate_sample(response: str) -> bool:
    """Validate model response format."""
    valid_labels = {"DOVISH", "HAWKISH", "NEUTRAL"}
    return response.strip().upper() in valid_labels


def load_fomc_dataset():
    """Load FOMC dataset with progress tracking."""
    dataset = safe_load_dataset(
        "gtfintechlab/fomc_communication", trust_remote_code=True
    )
    test_data = dataset["test"]  # type: ignore
    logger.debug(f"Loaded {len(test_data)} test samples")
    return test_data


def save_inference_results(
    df: pd.DataFrame, path: Path, metadata: Dict[str, Any]
) -> None:
    """Save results with metadata about the run."""
    metadata_path = path.with_suffix(".meta.json")
    metadata.update(
        {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(df),
            "successful_samples": len(df[df["llm_responses"].notna()]),
            "failed_samples": len(df[df["llm_responses"].isna()]),
        }
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    df.to_csv(path, index=False)
    logger.debug(f"Results and metadata saved to {path.parent}")


def fomc_inference(args):
    """Run FOMC inference with improved logging and error handling."""
    # Extract provider and model info
    model_parts = args.model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1]

    # Generate filename first
    results_path = generate_inference_filename("fomc", args.model)

    # Detailed startup logging - keep critical info at INFO level
    logger.info(f"Starting FOMC inference with {model_name}")
    logger.debug(f"Provider: {provider}")
    logger.debug(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.debug(f"Output directory: {results_path.parent}")
    logger.debug(f"Output filename: {results_path.name}")

    # Load dataset
    test_data = load_fomc_dataset()

    # Initialize result containers
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        fomc_prompt = get_prompt("fomc", PromptFormat.FEW_SHOT)
    else:
        fomc_prompt = get_prompt("fomc", PromptFormat.ZERO_SHOT)
    if fomc_prompt is None:
        raise RuntimeError("FOMC prompt not found in registry")

    # Get all sentences and labels
    all_sentences = [item["sentence"] for item in test_data]  # type: ignore
    all_labels = [item["label"] for item in test_data]  # type: ignore

    # Create batches
    sentence_batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(sentence_batches)
    logger.info(f"Processing {len(all_sentences)} samples in {total_batches} batches")

    # Process batches with progress bar
    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": fomc_prompt(sentence)}]
            for sentence in sentence_batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add None values for failed batch
            for _ in sentence_batch:
                sentences.append(None)
                complete_responses.append(None)
                llm_responses.append(None)
                actual_labels.append(None)
            continue

        # Process responses
        for sentence, response in zip(sentence_batch, batch_responses):
            sentences.append(sentence)
            complete_responses.append(response)
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = "Error"

            llm_responses.append(response_label)

            actual_labels.append(all_labels[len(llm_responses) - 1])

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    # Create results DataFrame
    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Log final statistics
    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
