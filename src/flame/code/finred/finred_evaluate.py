import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from flame.code.prompts.constants import (
    finred_extraction_labels as possible_relationships,
)
from flame.code.prompts.registry import PromptFormat, get_prompt
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.logging_utils import setup_logger

logger = setup_logger(
    name="finred_evaluation",
    log_file=LOG_DIR / "finred_evaluation.log",
    level=LOG_LEVEL,
)


def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")


def finred_evaluate(file_name, args):
    """Evaluate FinRED dataset and return results and metrics DataFrames."""
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "finred"
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Note: Path definition removed - evaluate.py handles saving

    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    correct_labels = df["actual_labels"].tolist()
    extracted_labels = []
    all_responses = df["llm_responses"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        extraction_prompt_func = get_prompt("finred", PromptFormat.EXTRACTION)
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(sentence)}]
            for sentence in sentence_batch
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in sentence_batch:
                extracted_labels.append("NO-REL")

        # Process responses
        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                extracted_label = "NO-REL"

            # Normalize and validate extracted label
            extracted_label = extracted_label.replace(" ", "")
            if extracted_label not in possible_relationships:
                logger.error(f"Invalid label: {extracted_label}")
                extracted_label = "NO-REL"

            extracted_labels.append(extracted_label)

    df["extracted_labels"] = extracted_labels

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
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1],
        }
    )

    # Note: Metrics saving removed - evaluate.py handles saving

    return df, metrics_df
