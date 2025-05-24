import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import LOG_DIR, LOG_LEVEL
from flame.code.prompts.registry import get_prompt, PromptFormat

# Setup logger
logger = setup_logger(
    name="numclaim_evaluation",
    log_file=LOG_DIR / "numclaim_evaluation.log",
    level=LOG_LEVEL,
)


# Mapping function to convert labels to binary
def map_labels(label):
    return 1 if str(label).upper() == "INCLAIM" else 0


def numclaim_evaluate(file_name, args):
    logger.info(f"Starting evaluation for Numclaim with model {args.model}...")
    # Load data from the specified file
    results_file = Path(file_name)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file {results_file} not found.")

    df = pd.read_csv(results_file)
    correct_labels = df["actual_labels"].apply(map_labels).tolist()
    llm_responses = df["llm_responses"].tolist()

    # Initialize the column for storing extracted labels if it doesn't exist
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    batch_size = args.batch_size
    indices = list(range(len(df)))
    index_batches = chunk_list(indices, batch_size)

    logger.info(f"Processing {len(df)} rows in {len(index_batches)} batches.")

    for batch_idx, batch_indices in enumerate(index_batches):
        llm_responses_batch = [llm_responses[i] for i in batch_indices]
        extraction_prompt_func = get_prompt("numclaim", PromptFormat.EXTRACTION)
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(llm_response)}]
            for llm_response in llm_responses_batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, len(index_batches)
            )

            for idx, (response, row_idx) in enumerate(
                zip(batch_responses, batch_indices)
            ):
                try:
                    extracted_label = response.choices[0].message.content.strip()  # type: ignore
                    mapped_extracted_label = map_labels(extracted_label)

                    # Update the DataFrame
                    df.at[row_idx, "extracted_labels"] = mapped_extracted_label

                except Exception as e:
                    logger.error(f"Error processing response for row {row_idx}: {e}")
                    df.at[row_idx, "extracted_labels"] = None

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            for row_idx in batch_indices:
                df.at[row_idx, "extracted_labels"] = None

    print(df["actual_labels"].value_counts())
    print(df["extracted_labels"].value_counts())

    # Calculate evaluation metrics
    extracted_labels = df["extracted_labels"].dropna().tolist()
    precision = precision_score(correct_labels, extracted_labels, average="binary")
    recall = recall_score(correct_labels, extracted_labels, average="binary")
    f1 = f1_score(correct_labels, extracted_labels, average="binary")
    accuracy = accuracy_score(correct_labels, extracted_labels)

    # Log the evaluation metrics
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Save evaluation metrics to DataFrame
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
            "Value": [precision, recall, f1, accuracy],
        }
    )

    logger.info("Evaluation completed.")

    return df, metrics_df
