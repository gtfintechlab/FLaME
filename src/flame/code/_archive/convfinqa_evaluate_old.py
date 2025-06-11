import re

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from flame.code.prompts.registry import PromptFormat, get_prompt
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.logging_utils import setup_logger

# Setup logger
logger = setup_logger(
    name="convfinqa_evaluation",
    log_file=LOG_DIR / "convfinqa_evaluation.log",
    level=LOG_LEVEL,
)


# Function to extract numerical values using regex
def extract_numerical_value(text):
    match = re.search(r"(\d+(\.\d+)?%?)", text)
    return match.group(0) if match else None


# Main evaluation function
def convfinqa_evaluate(file_name, args):
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "convfinqa"
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Note: Path definition removed - evaluate.py handles saving

    extraction_response = []
    extraction_model_response = []
    regex_extraction = []

    # Prepare all prompts for batch processing
    extraction_prompt_func = get_prompt("convfinqa", PromptFormat.EXTRACTION)
    all_responses = df["response"].tolist()

    logger.info(
        f"Processing {len(all_responses)} responses in batches of {args.batch_size}"
    )

    # Create batches
    batches = list(chunk_list(all_responses, args.batch_size))
    total_batches = len(batches)

    # Process batches with progress bar
    for batch_idx, batch_responses in enumerate(
        tqdm(batches, desc="Extracting ConvFinQA answers")
    ):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(response)}]
            for response in batch_responses
        ]

        try:
            # Process batch with retry logic
            batch_results = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            # Process each result in the batch
            for response in batch_results:
                extraction_model_response.append(response)
                try:
                    response_text = response.choices[0].message.content
                    extraction_response.append(response_text)

                    numerical_value = extract_numerical_value(response_text)
                    regex_extraction.append(numerical_value)
                except Exception as e:
                    logger.debug(f"Error extracting from response: {str(e)}")
                    extraction_response.append(None)
                    regex_extraction.append(None)

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add None values for failed batch
            for _ in batch_responses:
                extraction_model_response.append(None)
                extraction_response.append(None)
                regex_extraction.append(None)

    # Adding results to DataFrame
    df["extraction_model_response"] = extraction_model_response
    df["extraction_response"] = extraction_response
    df["regex_extraction"] = regex_extraction

    # Accuracy calculation
    correct_labels = df["actual_label"].tolist()
    predictions = [
        str(pred) if pd.notna(pred) else "Error" for pred in regex_extraction
    ]

    # Calculate metrics
    # Convert labels to strings for comparison
    correct_labels_str = [str(label) for label in correct_labels]

    accuracy = accuracy_score(correct_labels_str, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels_str, predictions, average="weighted", zero_division=0
    )

    # Log metrics
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

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}.")
    # Note: File saving removed - evaluate.py handles saving

    return df, metrics_df
