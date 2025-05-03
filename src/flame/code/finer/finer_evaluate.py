import pandas as pd
import numpy as np
import json
import re
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_DIR, LOG_LEVEL
from flame.code.extraction_prompts import finer_extraction_prompt
from tqdm import tqdm

logger = setup_logger(
    name="finer_evaluation",
    log_file=LOG_DIR / "finer_evaluation.log",
    level=LOG_LEVEL,
)


def clean_extracted_list(response: str) -> str:
    """Clean and format the extracted response into a valid JSON list."""
    cleaned_response = re.sub(r"[^\d,]", "", response)
    cleaned_response = re.sub(r"(\d)(\d)", r"\1,\2", cleaned_response)
    if not (cleaned_response.startswith("[") and cleaned_response.endswith("]")):
        cleaned_response = f"[{cleaned_response}]"
    return cleaned_response


def finer_evaluate(file_name, args):
    """
    Evaluate a Finer dataset row-by-row (list-of-lists) without flattening.
    Skip rows where the length of actual vs. predicted lists differ.
    Compute row-level metrics and aggregate them.
    """
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    correct_labels = df["actual_labels"].apply(
        lambda x: json.loads(x) if pd.notna(x) else []
    )

    extracted_labels = []

    all_responses = df["llm_responses"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    logger.info(f"Processing {len(df)} rows in {total_batches} batches.")
    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": finer_extraction_prompt(response)}]
            for response in batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            for _ in batch:
                extracted_labels.append([])
            continue

        for response in batch_responses:
            try:
                llm_response = response.choices[0].message.content.strip()  # type: ignore
                cleaned_response = clean_extracted_list(llm_response)
                extracted_tokens = json.loads(cleaned_response)
                extracted_labels.append(extracted_tokens)

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {llm_response}. Error: {e}")
                extracted_labels.append([])

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df["extracted_labels"] = extracted_labels

    row_precisions = []
    row_recalls = []
    row_f1s = []
    row_accuracies = []
    for i in range(len(correct_labels)):
        y_true = correct_labels[i]
        y_pred = extracted_labels[i]
        if len(y_true) != len(y_pred):
            logger.debug(
                f"Skipping row {i} because lengths differ (true={len(y_true)}, pred={len(y_pred)})."
            )
            continue

        try:
            p = precision_score(y_true, y_pred, average="macro", zero_division=0)
            r = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f = f1_score(y_true, y_pred, average="macro", zero_division=0)
            a = accuracy_score(y_true, y_pred)
            row_precisions.append(p)
            row_recalls.append(r)
            row_f1s.append(f)
            row_accuracies.append(a)
        except ValueError as e:
            logger.error(f"Skipping row {i} due to ValueError: {e}")
    if not row_precisions:
        logger.warning("No rows were evaluated (row_precisions is empty).")
        return None

    macro_precision = np.mean(row_precisions)
    macro_recall = np.mean(row_recalls)
    macro_f1 = np.mean(row_f1s)
    macro_accuracy = np.mean(row_accuracies)
    logger.info(f"Macro Precision: {macro_precision:.4f}")
    logger.info(f"Macro Recall: {macro_recall:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Macro Accuracy: {macro_accuracy:.4f}")
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
            "Value": [macro_precision, macro_recall, macro_f1, macro_accuracy],
        }
    )

    success_rate = df["extracted_labels"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
