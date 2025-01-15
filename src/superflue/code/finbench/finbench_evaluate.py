import pandas as pd
import logging
from datetime import date
from pathlib import Path
from litellm import completion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
import litellm
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

# Configure logging
logger = setup_logger(
    name="finbench_evaluation",
    log_file=LOG_DIR / "finbench_evaluation.log",
    level=LOG_LEVEL,
)

# Define label mapping
label_mapping = {
    "LOW RISK": 0,
    "HIGH RISK": 1,
}

def extraction_prompt(llm_response: str):
    """Generate a prompt for extracting risk labels."""
    prompt = f"""Based on the following list of labels: ‘HIGH RISK’, ‘LOW RISK’, extract the most relevant label from the following response:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation."""
    return prompt

def map_label_to_number(label: str):
    """Map the extracted label to its corresponding numerical value."""
    normalized_label = label.strip().upper()  # Normalize label to uppercase
    return label_mapping.get(normalized_label, -1)  # Return -1 if the label is not found

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def process_batch_with_retry(args, messages_batch, batch_idx, total_batches):
    """Process a batch with litellm's retry mechanism."""
    try:
        # Using litellm's built-in retry mechanism
        batch_responses = litellm.batch_completion(
            model=args.model,
            messages=messages_batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k else None,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_retries=3  # Using litellm's retry mechanism
        )
        logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses
            
    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise

def finbench_evaluate(file_name, args):
    """Evaluate the FinBench dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths for results and metrics
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    # Initialize extracted labels
    extracted_labels = []
    correct_labels = df["y"].tolist()
    all_responses = df["llm_responses"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": extraction_prompt(response)}]
            for response in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            # Process responses
            for response in batch_responses:
                extracted_label = response.choices[0].message.content.strip()  # type: ignore
                mapped_label = map_label_to_number(extracted_label)

                if mapped_label == -1:
                    logger.error(f"Invalid label for response {batch_idx}: {response}")
                else:
                    logger.info(f"Extracted label for row {batch_idx}: {mapped_label}")

                extracted_labels.append(mapped_label)

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in batch:
                extracted_labels.append(-1)

    df["extracted_labels"] = extracted_labels

    # Evaluate metrics
    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(correct_labels, extracted_labels, average="weighted")

    logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
    })

    # Save metrics to CSV
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
