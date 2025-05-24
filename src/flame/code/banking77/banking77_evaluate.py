import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import LOG_DIR, LOG_LEVEL
from flame.code.prompts.registry import get_prompt, PromptFormat
from tqdm import tqdm
from flame.code.prompts.constants import banking77_list

# Configure logging
logger = setup_logger(
    name="banking77_evaluate",
    log_file=LOG_DIR / "banking77_evaluate.log",
    level=LOG_LEVEL,
)

# Banking 77 categories list and mappings
banking77_label_map = {category: index for index, category in enumerate(banking77_list)}


def map_extracted_label_to_number(extracted_label: str):
    """Map the extracted label to its corresponding numerical value."""
    # Handle special "NO_MATCH" case from extraction
    if extracted_label == "NO_MATCH":
        logger.debug("Label extraction returned NO_MATCH")
        return -1

    # Clean up the extracted label by stripping whitespace and handling common variations
    cleaned_label = extracted_label.strip()

    if cleaned_label not in banking77_label_map:
        logger.error(f"Label not found: {repr(cleaned_label)}")
        logger.debug(f"Available labels: {list(banking77_label_map.keys())}")
    return banking77_label_map.get(
        cleaned_label, -1
    )  # Return -1 if the label is not found


def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")


def banking77_evaluate(file_name, args):
    """Evaluate Banking 77 results and return results and metrics DataFrames."""
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "banking77"
    if hasattr(args, "model"):
        logger.info(f"Starting evaluation for {task} using model {args.model}.")
    else:
        logger.info(f"Starting evaluation for {task}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Note: Path definition removed - evaluate.py handles saving

    # Initialize extracted_labels column if it doesn't exist
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    extracted_labels = []
    all_responses = df["llm_responses"].tolist()
    correct_labels = df["actual_labels"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch
        extraction_prompt_func = get_prompt("banking77", PromptFormat.EXTRACTION)
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
            # Add None values for failed batch
            for _ in batch:
                extracted_labels.append(-1)

        # Process responses
        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                extracted_label = "Error"
            # print(extracted_label)
            mapped_label = map_extracted_label_to_number(extracted_label)

            if mapped_label == -1:
                logger.debug(f"Error processing response {batch_idx}: {response}")

            extracted_labels.append(mapped_label)
            logger.debug(f"Processed {len(extracted_labels)}/{len(df)} responses.")

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df["extracted_labels"] = extracted_labels
    # Evaluate performance
    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, extracted_labels, average="weighted"
    )

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

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
