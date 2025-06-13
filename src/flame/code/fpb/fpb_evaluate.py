import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from flame.code.prompts.registry import PromptFormat, get_prompt
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.logging_utils import get_component_logger

# Configure logging
logger = get_component_logger("evaluation", "fpb")

# Define label mapping
label_mapping = {
    "NEUTRAL": 1,
    "NEGATIVE": 0,
    "POSITIVE": 2,
}


def map_label_to_number(label: str):
    """Map the extracted label to its corresponding numerical value after normalizing."""
    normalized_label = label.strip().upper()  # Normalize label to uppercase
    return label_mapping.get(
        normalized_label, -1
    )  # Return -1 if the label is not found


def fpb_evaluate(file_name, args):
    """Evaluate FPB dataset and return results and metrics DataFrames."""
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "fpb"
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file with the LLM responses
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Note: Path definition removed - evaluate.py handles saving

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
        extraction_prompt_func = get_prompt("fpb", PromptFormat.EXTRACTION)
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(response)}]
            for response in batch_content
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(batch_content)):
                extracted_labels.append(-1)

        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()
                mapped_label = map_label_to_number(extracted_label)

                if mapped_label == -1:
                    logger.debug(f"Invalid label for response: {extracted_label}")

            except Exception as e:
                logger.error(f"Error extracting response: {e}")
                mapped_label = -1

            extracted_labels.append(mapped_label)

    # Update the dataframe with extracted labels
    df["extracted_labels"] = extracted_labels

    # Calculate metrics
    correct_labels_array = np.array(correct_labels)
    extracted_labels_array = np.array(extracted_labels)
    accuracy = accuracy_score(correct_labels_array, extracted_labels_array)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels_array, extracted_labels_array, average="weighted"
    )

    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1],
        }
    )

    return df, metrics_df
