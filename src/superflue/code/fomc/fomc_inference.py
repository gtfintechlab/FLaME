"""FOMC inference module."""

from datetime import datetime
import pandas as pd
from superflue.code.prompts import fomc_prompt
from superflue.utils.save_utils import save_inference_results
from superflue.utils.batch_utils import (
    chunk_list,
    process_batch_with_retry,
    process_batch_responses,
)
from superflue.utils.logging_utils import get_logger
from superflue.utils.sampling_utils import load_and_sample_dataset

# Get logger for this module
logger = get_logger(__name__)


def validate_sample(response: str) -> bool:
    """Validate model response format."""
    valid_labels = {"DOVISH", "HAWKISH", "NEUTRAL"}
    return any(label in response.strip().upper() for label in valid_labels)


def fomc_inference(args):
    """Run inference on FOMC test dataset."""
    # Load and optionally sample test data
    test_data = load_and_sample_dataset(
        dataset_path=f"{args.dataset_org}/fomc_communication",
        sample_size=args.sample_size,
        sample_method=args.sample_method,
        trust_remote_code=True,
    )

    # Extract sentences and labels
    all_sentences = test_data["sentence"]
    all_labels = test_data["label"]

    # Initialize result lists
    llm_responses = []
    complete_responses = []
    actual_label = []
    sentences = []

    # Create batches
    sentence_batches = chunk_list(all_sentences, args.batch_size)
    label_batches = chunk_list(all_labels, args.batch_size)
    total_batches = len(sentence_batches)

    logger.info(f"Starting inference with {total_batches} batches")
    logger.debug(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Process each batch
    for batch_idx, (sentence_batch, label_batch) in enumerate(
        zip(sentence_batches, label_batches)
    ):
        # Prepare messages for the batch
        messages_batch = [
            [{"role": "user", "content": fomc_prompt(sentence=sentence)}]
            for sentence in sentence_batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args=args,
                messages_batch=messages_batch,
                batch_idx=batch_idx,
                total_batches=total_batches,
            )

            # Process and validate responses
            (
                batch_llm_responses,
                batch_complete_responses,
                batch_labels,
                batch_sentences,
            ) = process_batch_responses(
                batch_responses=batch_responses,
                validator_fn=validate_sample,
                logger=logger,
                batch_idx=batch_idx,
                labels=label_batch,
                sentences=sentence_batch,
            )

            # Extend result lists
            llm_responses.extend(batch_llm_responses)
            complete_responses.extend(batch_complete_responses)
            actual_label.extend(batch_labels)
            sentences.extend(batch_sentences)

        except Exception as e:
            logger.error(f"Failed to process batch {batch_idx + 1}: {str(e)}")
            # Add None values for failed batch
            for _ in range(len(sentence_batch)):
                llm_responses.append(None)
                complete_responses.append(None)
                actual_label.append(None)
                sentences.append(None)
            continue

    # Create DataFrame with results
    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_label": actual_label,
            "complete_responses": complete_responses,
        }
    )

    # Log final statistics
    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")
    logger.debug(f"Total successful responses: {df['llm_responses'].notna().sum()}")
    logger.debug(f"Total failed responses: {df['llm_responses'].isna().sum()}")
    logger.debug(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Extract provider and model info for metadata
    model_parts = args.inference_model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1]

    # Save results with metadata
    metadata = {
        "model": args.inference_model,
        "provider": provider,
        "model_name": model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "repetition_penalty": args.repetition_penalty,
        "dataset_org": args.dataset_org,
        "success_rate": success_rate,
    }

    # Use our save utility
    save_inference_results(
        df=df, task="fomc", model=args.inference_model, metadata=metadata
    )

    return df
