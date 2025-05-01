from typing import Dict, Tuple, List, Any
import pandas as pd
from datetime import datetime
from pathlib import Path
import uuid
from litellm import batch_completion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
import time
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
    """Map the extracted label to its corresponding numerical value after normalizing.

    Args:
        label: The text label to convert

    Returns:
        The numerical value corresponding to the label, or -1 if invalid
    """
    normalized_label = label.strip().upper()  # Normalize label to uppercase
    return label_mapping.get(
        normalized_label, -1
    )  # Return -1 if the label is not found


def save_progress(df: pd.DataFrame, path: Path) -> None:
    """Save the current progress to a CSV file.

    Args:
        df: DataFrame containing the evaluation results
        path: Path where the CSV should be saved
    """
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame has the required columns.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ["llm_responses", "actual_labels"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def generate_evaluation_filename(task: str, model: str) -> Tuple[str, Path]:
    """Generate a unique filename for evaluation results.

    Args:
        task: The task name (e.g., 'fomc')
        model: The full model path (e.g., 'together_ai/meta-llama/Llama-2-7b')

    Returns:
        Tuple of (base_filename, full_path)
    """
    # Extract provider and model name
    model_parts = model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1].replace("-", "_")  # Replace hyphens with underscores

    # Generate timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID

    # Construct base filename
    base_filename = f"{task}_{provider}_{model_name}_{timestamp}_{uid}"

    # Create full path
    full_path = EVALUATION_DIR / task / f"evaluation_{base_filename}.csv"
    full_path.parent.mkdir(parents=True, exist_ok=True)

    return base_filename, full_path


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_batch_with_retry(
    model: str,
    messages_batch: List[List[Dict]],
    args,
    batch_idx: int,
    total_batches: int,
):
    """Process a batch with litellm's retry mechanism."""
    try:
        batch_responses = batch_completion(
            model=model,
            messages=messages_batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_retries=3,
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

    # Generate unique filename and paths first for logging
    base_filename, evaluation_results_path = generate_evaluation_filename(
        task, args.model
    )

    # Extract provider and model name for logging
    model_parts = args.model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1]

    # Log detailed startup information
    logger.info(
        f"Starting {task} evaluation on model '{model_name}' from provider '{provider}'"
    )
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    relative_path = evaluation_results_path.relative_to(EVALUATION_DIR.parent)
    logger.info(f"Output directory: ./{relative_path.parent}")
    logger.info(f"Output filename: {relative_path.name}")

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
                    args.model, messages_batch, args, batch_idx, total_batches
                )

            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
                # Mark failed extractions with -1
                for idx in indices_batch:
                    df.at[idx, "extracted_labels"] = -1
                time.sleep(10.0)
                continue

            # Process responses
            for idx, response in zip(indices_batch, batch_responses):
                try:
                    extracted_label = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                    extracted_label = "Error"
                mapped_label = map_label_to_number(extracted_label)

                # Update DataFrame with the result
                df.at[idx, "extracted_labels"] = mapped_label

            # Save progress after each batch
            save_progress(df, evaluation_results_path)

            pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

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

    # Create metrics DataFrame with additional metadata
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1],
        }
    )

    # Save metrics DataFrame with consistent naming
    # metrics_path = evaluation_results_path.with_name(f"evaluation_{base_filename}_metrics.csv")
    # metrics_df.to_csv(metrics_path, index=False)
    # logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
