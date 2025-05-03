from typing import Dict, Tuple, List
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from flame.utils.logging_utils import setup_logger
from flame.code.extraction_prompts import fomc_extraction_prompt
from flame.utils.batch_utils import process_batch_with_retry, chunk_list
from flame.config import LOG_DIR, LOG_LEVEL
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


def map_label_to_number(label: str) -> int:
    normalized_label = label.strip().upper()  # Normalize label to uppercase
    return label_mapping.get(
        normalized_label, -1
    )  # Return -1 if the label is not found


def fomc_evaluate(file_name: str, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    df = pd.read_csv(file_name)

    correct_labels = df["actual_labels"].tolist()
    extracted_labels: List[int] = []

    all_responses = df["llm_responses"].tolist()
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    logger.info(f"Processing {len(df)} rows in {total_batches} batches.")
    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": fomc_extraction_prompt(response)}]
            for response in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args.model, messages_batch, args, batch_idx
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in batch:
                extracted_labels.append(-1)
            continue

        # Process responses
        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                extracted_label = "Error"
            mapped_label = map_label_to_number(extracted_label)
            extracted_labels.append(mapped_label)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    # Convert all extracted labels to list for metrics calculation
    df["extracted_labels"] = extracted_labels

    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, extracted_labels, average="weighted"
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

    success_rate = df["extracted_labels"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
