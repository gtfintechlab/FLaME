import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# litellm.set_verbose=True
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import LOG_DIR, LOG_LEVEL
from flame.code.prompts.registry import get_prompt, PromptFormat

# Set up logging
logger = setup_logger(
    name="causal_classification_evaluation",
    log_file=LOG_DIR / "causal_classification_evaluation.log",
    level=LOG_LEVEL,
)


def normalize_response(response):
    """Normalize the LLM response to extract the predicted label."""
    try:
        response = response.strip()
        if response.isdigit():
            return int(response)
        elif "0" in response:
            return 0
        elif "1" in response:
            return 1
        elif "2" in response:
            return 2
        else:
            raise ValueError(f"Invalid response format: {response}")
    except Exception as e:
        logger.error(f"Error normalizing response: {e}")
        return None


def causal_classification_evaluate(file_name, args):
    """Evaluate causal classification results with label extraction and comparison."""
    task = "causal_classification"
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    if "actual_labels" not in df.columns or "llm_responses" not in df.columns:
        logger.error(
            "The input CSV must contain 'actual_labels' and 'llm_responses' columns."
        )
        raise ValueError("Missing required columns in the input file.")
    batch_size = args.batch_size
    indices = list(range(len(df)))
    index_batches = chunk_list(indices, batch_size)

    extracted_labels = []
    metrics = []
    logger.info(f"Processing {len(df)} rows in {len(index_batches)} batches.")
    for batch_idx, batch_indices in enumerate(index_batches):
        llm_responses_batch = [df.at[i, "llm_responses"] for i in batch_indices]
        actual_labels_batch = [df.at[i, "actual_labels"] for i in batch_indices]
        logger.info(f"Processing batch {batch_idx + 1} with {len(batch_indices)} rows.")
        extraction_prompt_func = get_prompt(
            "causal_classification", PromptFormat.EXTRACTION
        )
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(llm_response)}]
            for llm_response in llm_responses_batch
        ]
        logger.info(f"Generated messages for batch {messages_batch}.")

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, len(index_batches)
            )
            logger.info(f"{batch_responses}")
            for idx, (response, actual_label) in enumerate(
                zip(batch_responses, actual_labels_batch)
            ):
                try:
                    llm_response = response.choices[0].message.content.strip()  # type: ignore
                    predicted_label = normalize_response(llm_response)

                    if predicted_label is not None:
                        extracted_labels.append(predicted_label)
                        metrics.append(
                            (actual_label, predicted_label)
                        )  # Store actual vs predicted pairs
                    else:
                        extracted_labels.append(None)

                except Exception as e:
                    logger.error(
                        f"Error processing response for row {batch_indices[idx]}: {e}"
                    )
                    extracted_labels.append(None)
                    metrics.append(
                        {
                            "precision": 0,
                            "recall": 0,
                            "f1": 0,
                            "accuracy": 0,
                        }
                    )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            extracted_labels.extend([None] * len(batch_indices))
            metrics.extend(
                [{"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}]
                * len(batch_indices)
            )
            continue

    # Add extracted labels to the DataFrame
    df["extracted_labels"] = extracted_labels

    # Aggregate metrics
    valid_indices = [
        i for i in range(len(extracted_labels)) if extracted_labels[i] is not None
    ]
    filtered_predicted = [extracted_labels[i] for i in valid_indices]
    filtered_actual = [df.at[i, "actual_labels"] for i in valid_indices]

    # Compute evaluation metrics
    precision = precision_score(filtered_actual, filtered_predicted, average="macro")
    recall = recall_score(filtered_actual, filtered_predicted, average="macro")
    f1 = f1_score(filtered_actual, filtered_predicted, average="macro")
    accuracy = accuracy_score(filtered_actual, filtered_predicted)

    # Metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
            "Value": [precision, recall, f1, accuracy],
        }
    )

    # Log metrics
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    return df, metrics_df
