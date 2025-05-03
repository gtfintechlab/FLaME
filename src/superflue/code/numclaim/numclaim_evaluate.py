import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
from superflue.utils.logging_utils import setup_logger
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.code.extraction_prompts import numclaim_extraction_prompt
from superflue.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm

logger = setup_logger(
    name="numclaim_evaluation",
    log_file=LOG_DIR / "numclaim_evaluation.log",
    level=LOG_LEVEL,
)


def map_labels(label):
    return 1 if str(label).upper() == "INCLAIM" else 0


def numclaim_evaluate(file_name, args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    results_file = Path(file_name)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file {results_file} not found.")

    df = pd.read_csv(results_file)
    correct_labels = df["actual_labels"].apply(map_labels).tolist()
    extracted_labels = []

    all_responses = df["llm_responses"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    logger.info(f"Processing {len(df)} rows in {total_batches} batches.")
    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": numclaim_extraction_prompt(response)}]
            for response in batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            for _ in batch:
                extracted_labels.append(None)
            continue

        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()  # type: ignore
                mapped_extracted_label = map_labels(extracted_label)
                extracted_labels.append(mapped_extracted_label)
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                extracted_labels.append(None)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df["extracted_labels"] = extracted_labels

    extracted_labels = df["extracted_labels"].dropna().tolist()
    precision = precision_score(correct_labels, extracted_labels, average="binary")
    recall = recall_score(correct_labels, extracted_labels, average="binary")
    f1 = f1_score(correct_labels, extracted_labels, average="binary")
    accuracy = accuracy_score(correct_labels, extracted_labels)

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    metrics_df = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
            "Value": [precision, recall, f1, accuracy],
        }
    )

    logger.info("Evaluation completed.")

    success_rate = df["extracted_labels"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
