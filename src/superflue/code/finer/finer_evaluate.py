import pandas as pd
import logging
import numpy as np
from datetime import date
from pathlib import Path
import json
import re
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from litellm import batch_completion
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
 
logger = setup_logger(
    name="finer_evaluation",
    log_file=LOG_DIR / "finer_evaluation.log",
    level=LOG_LEVEL,
)
 
def extraction_prompt_finer(llm_response: str):
    """Generate a prompt to extract numeric labels for named entity recognition."""
    prompt = f"""For each token in the following response, map the named entity labels to these numeric values:
                    - "O" (Other): 0
                    - "PER_B" (Person_B): 1
                    - "PER_I" (Person_I): 2
                    - "LOC_B" (Location_B): 3
                    - "LOC_I" (Location_I): 4
                    - "ORG_B" (Organisation_B): 5
                    - "ORG_I" (Organisation_I): 6
 
                Provide only the list of integer labels, in the format:
                [0, 1, 0, ...]
 
                Do not include any additional text, explanations, or formatting other than a plain list.
 
                LLM response:
                "{llm_response}"."""
    return prompt
 
def clean_extracted_list(response: str) -> str:
    """Clean and format the extracted response into a valid JSON list."""
    cleaned_response = re.sub(r"[^\d,]", "", response)
    cleaned_response = re.sub(r"(\d)(\d)", r"\1,\2", cleaned_response)
    if not (cleaned_response.startswith("[") and cleaned_response.endswith("]")):
        cleaned_response = f"[{cleaned_response}]"
    return cleaned_response
 
def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def finer_evaluate(file_name, args):
    """
    Evaluate a Finer dataset row-by-row (list-of-lists) without flattening.
    Skip rows where the length of actual vs. predicted lists differ.
    Compute row-level metrics and aggregate them.
    """
    task = args.dataset.strip('“”"')
    logger.info(f"Starting row-by-row evaluation for {task} using model {args.model}.")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    if "actual_labels" not in df.columns or "llm_responses" not in df.columns:
        logger.error("The input CSV must contain 'actual_labels' and 'llm_responses' columns.")
        raise ValueError("Missing required columns in the input file.")

    correct_labels = df["actual_labels"].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    
    extracted_labels = []

    batch_size = args.batch_size
    indices = list(range(len(df)))
    index_batches = chunk_list(indices, batch_size)
    logger.info(f"Processing {len(df)} rows in {len(index_batches)} batches.")
    for batch_idx, batch_indices in enumerate(index_batches):
        llm_responses_batch = [df.at[i, "llm_responses"] for i in batch_indices]
        logger.info(f"Processing batch {batch_idx + 1} with {len(batch_indices)} rows.")
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_finer(llm_response)}]
            for llm_response in llm_responses_batch
        ]
        try:
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, len(index_batches))
            logger.info(f"Processed responses for batch {batch_idx + 1}.")
            for idx, (response, row_idx) in enumerate(zip(batch_responses, batch_indices)):
                try:
                    if response is None or not hasattr(response, "choices") or not response.choices:
                        raise ValueError(f"Invalid API response: {response}")
                    llm_response = response.choices[0].message.content.strip()  # type: ignore
                    cleaned_response = clean_extracted_list(llm_response)
                    extracted_tokens = json.loads(cleaned_response)
                    extracted_labels.append(extracted_tokens)

                except Exception as e:
                    logger.error(f"Error processing response for row {row_idx}: {e}")
                    extracted_labels.append([])

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            extracted_labels.extend([[]] * len(batch_indices))
            continue
        
    row_precisions = []
    row_recalls = []
    row_f1s = []
    row_accuracies = []
    for i in range(len(correct_labels)):
        y_true = correct_labels[i]
        y_pred = extracted_labels[i]
        if len(y_true) != len(y_pred):
            logger.debug(f"Skipping row {i} because lengths differ (true={len(y_true)}, pred={len(y_pred)}).")
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
    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
        "Value": [macro_precision, macro_recall, macro_f1, macro_accuracy]
    })

    return df, metrics_df
