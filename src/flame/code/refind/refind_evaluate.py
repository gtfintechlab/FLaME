import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from flame.code.prompts.constants import (
    refind_possible_relationships as possible_relationships,
)
from flame.code.prompts.registry import PromptFormat, get_prompt
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.logging_utils import get_component_logger

logger = get_component_logger("evaluation", "refind")


def refind_evaluate(file_name, args):
    """Evaluate Refind dataset and return results and metrics DataFrames."""
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "refind"
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    # Load the CSV file with the LLM responses
    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Prepare extracted labels
    extracted_labels = []
    correct_labels = df["actual_labels"].tolist()
    all_responses = df["llm_responses"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    # Note: Path definition removed - evaluate.py handles saving

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        extraction_prompt_func = get_prompt("refind", PromptFormat.EXTRACTION)
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(llm_response)}]
            for llm_response in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            for _ in batch:
                extracted_labels.append("NO-REL")

        # Process responses
        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                extracted_labels.append("NO-REL")
            extracted_label = (
                extracted_label.replace(" ", "")
                .replace("/", "-")
                .replace("_", "-")
                .upper()
            )
            if extracted_label not in possible_relationships:
                logger.debug(f"Invalid label: {extracted_label}")
                extracted_label = "NO-REL"
            extracted_labels.append(extracted_label)

    df["extracted_labels"] = extracted_labels

    correct_labels = [
        label.replace(" ", "").replace("/", "-").replace("_", "-").upper()
        for label in correct_labels
    ]

    # Evaluate the performance
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

    # Note: Metrics saving removed - evaluate.py handles saving

    return df, metrics_df
