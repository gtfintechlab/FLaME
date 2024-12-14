"""MMLU evaluation module."""

from datetime import datetime
from pathlib import Path
import uuid
from typing import Optional, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score
from superflue.config import EVALUATION_DIR
from superflue.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


def extract_answer(llm_response: str) -> Optional[str]:
    """Extract the answer letter from the LLM response.

    Args:
        llm_response: Raw response from the language model

    Returns:
        Extracted answer letter or None if invalid
    """
    # Clean and normalize the response
    cleaned = llm_response.strip().upper()

    # If the response is a single letter A-D, return it
    if len(cleaned) == 1 and cleaned in ["A", "B", "C", "D"]:
        return cleaned

    # Look for exact phrases like "the answer is X" or "X is correct"
    for letter in ["A", "B", "C", "D"]:
        if any(
            phrase in cleaned
            for phrase in [
                f"ANSWER IS {letter}",
                f"THE ANSWER IS {letter}",
                f"{letter} IS CORRECT",
                f"{letter}.",
                f"ANSWER: {letter}",
            ]
        ):
            return letter

    return None


def save_progress(df: pd.DataFrame, path: Path) -> None:
    """Save the current evaluation progress.

    Args:
        df: DataFrame containing evaluation results
        path: Path to save the results
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
    required_columns = ["raw_response", "actual_answer", "subject"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def generate_evaluation_filename(task: str, model: str) -> Tuple[str, Path]:
    """Generate unique filename for evaluation results.

    Args:
        task: Task name (e.g., 'mmlu')
        model: Full model path

    Returns:
        Tuple of (base_filename, full_path)
    """
    # Extract provider and model name
    model_parts = model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1].replace("-", "_")

    # Generate timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]

    # Construct filename
    base_filename = f"{task}_{provider}_{model_name}_{timestamp}_{uid}"

    # Create full path
    full_path = EVALUATION_DIR / task / f"evaluation_{base_filename}.csv"
    full_path.parent.mkdir(parents=True, exist_ok=True)

    return base_filename, full_path


def mmlu_evaluate(file_name: str, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate MMLU inference results and calculate metrics.

    This function:
    1. Loads saved inference results
    2. Extracts predicted answers from raw responses
    3. Calculates overall and per-subject accuracy
    4. Saves detailed evaluation results and metrics

    Args:
        file_name: Path to CSV file containing inference results with columns:
            - raw_response: Complete model response
            - actual_answer: Correct answer (A/B/C/D)
            - subject: Question subject area
        args: Arguments containing:
            - model: Model identifier for logging
            - mmlu_subjects: List of subjects evaluated
            - mmlu_split: Dataset split used

    Returns:
        Tuple containing:
            - results_df: DataFrame with predictions and evaluation results
            - metrics_df: DataFrame with accuracy metrics per subject

    Raises:
        ValueError: If input file is missing required columns
        ValueError: If no valid predictions are found

    Example:
        >>> args = argparse.Namespace(
        ...     model="together_ai/meta-llama/Llama-2-7b",
        ...     mmlu_subjects=["high_school_microeconomics"],
        ...     mmlu_split="test"
        ... )
        >>> results_df, metrics_df = mmlu_evaluate("inference_results.csv", args)
        >>> print(f"Overall accuracy: {metrics_df.loc[0, 'Value']:.2f}")
    """
    task = args.dataset.strip('"""')

    # Generate unique filename and paths
    base_filename, evaluation_results_path = generate_evaluation_filename(
        task, args.model
    )

    # Extract provider and model info for logging
    model_parts = args.model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1]

    # Log startup information
    logger.info(
        f"Starting MMLU evaluation on model '{model_name}' from provider '{provider}'"
    )
    logger.info(f"Subjects: {args.mmlu_subjects or 'default economics subjects'}")
    logger.info(f"Split: {args.mmlu_split}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load and validate inference results
    try:
        df = pd.read_csv(file_name)
        logger.info(f"Loaded {len(df)} rows from {file_name}")
        validate_input_data(df)
    except Exception as e:
        logger.error(f"Error loading or validating input file: {e}")
        raise

    # Initialize predictions column if not present
    if "predicted_answer" not in df.columns:
        df["predicted_answer"] = None

    # Process each response
    for idx in range(len(df)):
        # Skip if already processed
        if pd.notna(df.at[idx, "predicted_answer"]):
            continue

        try:
            # Extract answer from raw response
            raw_response = df.at[idx, "raw_response"]
            predicted_answer = extract_answer(raw_response)

            # Update DataFrame with prediction
            df.at[idx, "predicted_answer"] = (
                predicted_answer if predicted_answer else pd.NA
            )

            if predicted_answer:
                logger.info(
                    f"Question {idx}: Predicted {predicted_answer}, Actual {df.at[idx, 'actual_answer']}"
                )
            else:
                logger.warning(
                    f"Question {idx}: Could not extract valid answer from response: {raw_response}"
                )

            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {idx}: {e}")
            continue

    # Calculate metrics
    valid_mask = df["predicted_answer"].notna()
    if not valid_mask.any():
        raise ValueError("No valid predictions for evaluation")

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(
        df.loc[valid_mask, "actual_answer"], df.loc[valid_mask, "predicted_answer"]
    )

    # Create metrics DataFrame
    metrics = []

    # Add overall accuracy
    metrics.append(
        {
            "Metric": "Accuracy",
            "Value": overall_accuracy,
            "Subject": "Overall",
            "Questions": valid_mask.sum(),
        }
    )

    # Add per-subject accuracy
    for subject in df["subject"].unique():
        subject_mask = (df["subject"] == subject) & valid_mask
        if subject_mask.any():
            subject_accuracy = accuracy_score(
                df.loc[subject_mask, "actual_answer"],
                df.loc[subject_mask, "predicted_answer"],
            )
            metrics.append(
                {
                    "Metric": "Accuracy",
                    "Value": subject_accuracy,
                    "Subject": subject,
                    "Questions": subject_mask.sum(),
                }
            )

    metrics_df = pd.DataFrame(metrics)

    # Log metrics
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    for _, row in metrics_df.iterrows():
        logger.info(
            f"{row['Subject']}: {row['Value']:.4f} ({row['Questions']} questions)"
        )

    # Save metrics
    metrics_path = evaluation_results_path.with_name(
        f"evaluation_{base_filename}_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
