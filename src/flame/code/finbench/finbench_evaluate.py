import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from flame.code.prompts.registry import PromptFormat, get_prompt
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.logging_utils import setup_logger

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


def map_label_to_number(label: str):
    """Map the extracted label to its corresponding numerical value."""
    normalized_label = label.strip().upper()  # Normalize label to uppercase
    return label_mapping.get(
        normalized_label, -1
    )  # Return -1 if the label is not found


def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")


def finbench_evaluate(file_name, args):
    """Evaluate the FinBench dataset and return results and metrics DataFrames."""
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "finbench"
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Note: Path definition removed - evaluate.py handles saving

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
        extraction_prompt_func = get_prompt("finbench", PromptFormat.EXTRACTION)
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(response)}]
            for response in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in batch:
                extracted_labels.append(-1)

        # Process responses
        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                extracted_label = "Error"
            mapped_label = map_label_to_number(extracted_label)

            if mapped_label == -1:
                logger.error(f"Invalid label for response {batch_idx}: {response}")
            else:
                logger.debug(f"Extracted label for row {batch_idx}: {mapped_label}")

            extracted_labels.append(mapped_label)

    df["extracted_labels"] = extracted_labels

    # Evaluate metrics
    correct_labels_array = np.array(correct_labels)
    extracted_labels_array = np.array(extracted_labels)
    accuracy = accuracy_score(correct_labels_array, extracted_labels_array)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels_array, extracted_labels_array, average="weighted"
    )

    logger.info(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1],
        }
    )

    # Note: Metrics saving removed - evaluate.py handles saving

    return df, metrics_df
