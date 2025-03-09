import pandas as pd
import logging
from datetime import date
from pathlib import Path
from litellm import completion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list
from superflue.utils.logging_utils import setup_logger
from superflue.code.extraction_prompts import fpb_extraction_prompt
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
import time
import litellm
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

# Configure logging
logger = setup_logger(
    name="fpb_evaluation",
    log_file=LOG_DIR / "fpb_evaluation.log",
    level=LOG_LEVEL,
)

# Define label mapping
label_mapping = {
    "NEUTRAL": 1,
    "NEGATIVE": 0,
    "POSITIVE": 2,
}

def map_label_to_number(label: str):
    """Map the extracted label to its corresponding numerical value after normalizing."""
    normalized_label = label.strip().upper()  # Normalize label to uppercase
    return label_mapping.get(normalized_label, -1)  # Return -1 if the label is not found

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def fpb_evaluate(file_name, args):
    """Evaluate FPB dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file with the LLM responses
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths for saving results
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize extracted labels if not present
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    correct_labels = df["actual_labels"].tolist()
    extracted_labels = []
    all_responses = df["llm_responses"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch_content in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": fpb_extraction_prompt(response)}]
            for response in batch_content
        ]
        try:
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(batch_content)):
                extracted_labels.append(-1)
        
        for response in batch_responses:
            try: 
                extracted_label = response.choices[0].message.content.strip()
                mapped_label = map_label_to_number(extracted_label)

                if mapped_label == -1:
                    logger.error(f"Invalid label for response: {extracted_label}")
            
            except Exception as e:
                logger.error(f"Error extracting response: {e}")
                mapped_label = -1

            extracted_labels.append(mapped_label)

    # Calculate metrics
    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, extracted_labels, average="weighted"
    )

    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1],
    })

    # # Save metrics DataFrame
    # metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    # metrics_df.to_csv(metrics_path, index=False)
    # logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
