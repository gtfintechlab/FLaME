from typing import Dict, Tuple, List
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
from litellm import completion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.save_utils import save_evaluation_results
from superflue.utils.path_utils import extract_model_from_inference_path
from superflue.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm

# Configure logging
logger = setup_logger(
    name="fomc_evaluation",
    log_file=LOG_DIR / "fomc_evaluation.log",
    level=LOG_LEVEL,
)

# Mapping of FOMC sentiment labels to numerical values
label_mapping: Dict[str, int] = {
    "DOVISH": 0,  # Indicates accommodative monetary policy stance
    "HAWKISH": 1,  # Indicates restrictive monetary policy stance
    "NEUTRAL": 2,  # Indicates balanced monetary policy stance
}


def extraction_prompt(llm_response: str) -> str:
    """Generate a prompt to extract the classification label from the LLM response.

    Args:
        llm_response: The raw response from the language model

    Returns:
        A formatted prompt string for label extraction
    """
    prompt = f"""Extract the classification label from the following LLM response. The label should be one of the following: 'HAWKISH', 'DOVISH', or 'NEUTRAL'.
                
                Here is the LLM response to analyze:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation."""
    return prompt


def map_label_to_number(label: str) -> int:
    """Map a text label to its corresponding numerical value.

    Args:
        label: The text label to map

    Returns:
        The corresponding numerical value, or -1 if invalid
    """
    return label_mapping.get(label.strip().upper(), -1)


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate the structure of input data.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required_columns = {"llm_responses", "actual_label"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_batch_with_retry(
    model: str, messages_batch: List, args, batch_idx: int, total_batches: int
) -> List:
    """Process a batch with litellm's retry mechanism."""
    try:
        # Using litellm's built-in retry mechanism
        batch_responses = completion(
            model=args.extraction_model,  # Use extraction_model for evaluation
            messages=messages_batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k else None,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_retries=3,  # Using litellm's retry mechanism
        )
        logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses

    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise


def fomc_evaluate(file_name: str, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate FOMC dataset and return results and metrics DataFrames.

    Args:
        file_name: Path to the CSV file containing LLM responses
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

    # Log detailed startup information
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

    correct_labels = df["actual_label"].tolist()
    extracted_labels: List[int] = []

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
                batch_responses = process_batch_with_retry(
                    args.extraction_model,
                    messages_batch,
                    args,
                    batch_idx,
                    total_batches,
                )

                # Process responses
                for idx, response in zip(indices_batch, batch_responses):
                    extracted_label = response.choices[0].message.content.strip()
                    mapped_label = map_label_to_number(extracted_label)

                    # Update DataFrame with the result
                    df.at[idx, "extracted_labels"] = mapped_label

                pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
                # Mark failed extractions with -1
                for idx in indices_batch:
                    df.at[idx, "extracted_labels"] = -1
                time.sleep(10.0)
                continue

    # Convert all extracted labels to list for metrics calculation
    extracted_labels = df["extracted_labels"].tolist()

    # Calculate metrics
    valid_indices = [i for i, label in enumerate(extracted_labels) if label != -1]
    if not valid_indices:
        logger.error("No valid labels extracted for evaluation")
        raise ValueError("No valid labels for evaluation")

    valid_extracted = [extracted_labels[i] for i in valid_indices]
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

    # Create metrics DataFrame
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

    # Use our new save utility
    save_evaluation_results(
        df=df,
        task=task,
        inference_model=inference_model,
        extraction_model=args.extraction_model,
        metadata=metadata,
    )

    # Create metrics DataFrame for return
    metrics_df = pd.DataFrame(
        {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
    )

    return df, metrics_df
