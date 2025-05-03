import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR, LOG_LEVEL
from superflue.code.extraction_prompts import (
    banking_77_extraction_prompt,
    banking77_label_map,
)
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from tqdm import tqdm

logger = setup_logger(
    name="banking77_evaluate",
    log_file=LOG_DIR / "banking77_evaluate.log",
    level=LOG_LEVEL,
)


def map_extracted_label_to_number(extracted_label: str):
    """Map the extracted label to its corresponding numerical value."""
    if extracted_label not in banking77_label_map:
        logger.error(f"Label not found: {extracted_label}")
    return banking77_label_map.get(extracted_label, -1)


def banking77_evaluate(file_name, args):
    """Evaluate Banking 77 results and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    extracted_labels = []
    all_responses = df["llm_responses"].tolist()
    correct_labels = df["actual_labels"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": banking_77_extraction_prompt(response)}]
            for response in batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in batch:
                extracted_labels.append(-1)
            continue

        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                extracted_label = "Error"
            mapped_label = map_extracted_label_to_number(extracted_label)

            if mapped_label == -1:
                logger.debug(f"Error processing response {batch_idx}: {response}")

            extracted_labels.append(mapped_label)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df["extracted_labels"] = extracted_labels
    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, extracted_labels, average="weighted"
    )

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    metrics_df = pd.DataFrame(
        {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1],
        }
    )

    success_rate = df["extracted_labels"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
