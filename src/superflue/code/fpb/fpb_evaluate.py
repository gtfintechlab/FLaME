from typing import Dict, Tuple
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import litellm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from superflue.utils.save_utils import save_evaluation_results
from superflue.utils.batch_utils import chunk_list
from superflue.utils.path_utils import extract_model_from_inference_path
from tqdm import tqdm
from superflue.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Define label mapping
label_mapping: Dict[str, int] = {
    "NEUTRAL": 1,
    "NEGATIVE": 0,
    "POSITIVE": 2,
}


def extraction_prompt(llm_response: str) -> str:
    """Generate a prompt to extract the most relevant label from the LLM response."""
    prompt = f"""Based on the following list of labels: 'NEGATIVE', 'POSITIVE', or 'NEUTRAL', extract the most relevant label from the following response:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation."""
    return prompt


def map_label_to_number(label: str) -> int:
    """Map text label to numerical value."""
    return label_mapping.get(label.strip().upper(), -1)


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate the structure of input data.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required_columns = {"llm_responses", "actual_labels"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")


def fpb_evaluate(file_name: str, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate FPB dataset using batching.

    Args:
        file_name: Path to the inference results file
        args: Arguments containing model configuration

    Returns:
        Tuple containing (results DataFrame, metrics DataFrame)

    Raises:
        ValueError: If input data validation fails
    """
    task = args.dataset.strip('"""')

    # Get inference model from input file
    inference_model = extract_model_from_inference_path(Path(file_name))
    if not inference_model:
        raise ValueError(
            f"Could not extract inference model from filename: {file_name}"
        )

    # Log startup information
    logger.info(f"Starting {task} evaluation")
    logger.info(f"Inference model: {inference_model}")
    logger.info(f"Extraction model: {args.extraction_model}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load and validate the CSV file
    try:
        df = pd.read_csv(file_name)
        logger.info(f"Loaded {len(df)} rows from {file_name}.")
        validate_input_data(df)
    except Exception as e:
        logger.error(f"Error loading or validating input file: {e}")
        raise

    # Initialize extracted labels if not present
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    correct_labels = df["actual_labels"].tolist()

    # Get indices of responses that need processing
    pending_indices = [
        i for i, label in enumerate(df["extracted_labels"]) if pd.isna(label)
    ]

    if pending_indices:
        # Create batches of pending responses
        pending_responses = [df["llm_responses"].iloc[i] for i in pending_indices]
        response_batches = chunk_list(pending_responses, args.batch_size)
        batch_indices = chunk_list(pending_indices, args.batch_size)
        total_batches = len(response_batches)

        logger.info(
            f"Processing {len(pending_indices)} responses in {total_batches} batches"
        )

        # Process batches with progress bar
        pbar = tqdm(
            zip(response_batches, batch_indices),
            total=total_batches,
            desc="Processing batches",
        )

        for batch_idx, (response_batch, indices_batch) in enumerate(pbar):
            # Prepare messages for batch
            messages_batch = [
                [{"role": "user", "content": extraction_prompt(response)}]
                for response in response_batch
            ]

            try:
                # Process batch with retry logic
                batch_responses = litellm.batch_completion(
                    model=args.extraction_model,
                    messages=messages_batch,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k if args.top_k else None,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    num_retries=3,
                )
                logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")

                # Process responses
                for idx, response in zip(indices_batch, batch_responses):
                    extracted_label = response.choices[0].message.content.strip()
                    mapped_label = map_label_to_number(extracted_label)
                    df.at[idx, "extracted_labels"] = mapped_label

                pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
                # Mark failed extractions with -1
                for idx in indices_batch:
                    df.at[idx, "extracted_labels"] = -1
                time.sleep(10.0)
                continue

    # Calculate metrics
    valid_indices = [
        i
        for i, label in enumerate(df["extracted_labels"])
        if label != -1 and not pd.isna(label)
    ]

    if not valid_indices:
        logger.error("No valid labels extracted for evaluation")
        raise ValueError("No valid labels for evaluation")

    valid_extracted = [df["extracted_labels"].iloc[i] for i in valid_indices]
    valid_correct = [correct_labels[i] for i in valid_indices]

    accuracy = accuracy_score(valid_correct, valid_extracted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_correct, valid_extracted, average="weighted"
    )

    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": len(df),
        "valid_samples": len(valid_indices),
        "failed_samples": len(df) - len(valid_indices),
    }

    # Save results with metadata
    metadata = {
        "inference_model": inference_model,
        "extraction_model": args.extraction_model,
        "metrics": metrics,
        "parameters": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "repetition_penalty": args.repetition_penalty,
        },
    }

    save_evaluation_results(
        df=df,
        task="fpb",
        inference_model=inference_model,
        extraction_model=args.extraction_model,
        metadata=metadata,
    )

    # Create metrics DataFrame for return
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1],
        }
    )

    return df, metrics_df
