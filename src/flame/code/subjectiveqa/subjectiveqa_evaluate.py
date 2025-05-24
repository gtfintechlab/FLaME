import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import LOG_DIR, LOG_LEVEL
from flame.code.prompts.registry import get_prompt, PromptFormat

# Setup logger
logger = setup_logger(
    name="subjectiveqa_evaluation",
    log_file=LOG_DIR / "subjectiveqa_evaluation.log",
    level=LOG_LEVEL,
)


def normalize_response(response):
    """Normalize the LLM response to extract the predicted label."""
    try:
        # Strip whitespace
        response = str(response).strip()

        # If the response is directly a valid number, return it
        if response.isdigit() and int(response) in [0, 1, 2]:
            return int(response)

        # Handle cases where the response contains extra text
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


def subjectiveqa_evaluate(file_name, args):
    """Evaluate SubjectiveQA results with extraction and batching logic."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    # Load the input CSV file
    data = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Define label pairs for evaluation
    label_pairs = [
        ("RELEVANT_actual_label", "RELEVANT_response"),
        ("SPECIFIC_actual_label", "SPECIFIC_response"),
        ("CAUTIOUS_actual_label", "CAUTIOUS_response"),
        ("ASSERTIVE_actual_label", "ASSERTIVE_response"),
        ("CLEAR_actual_label", "CLEAR_response"),
        ("OPTIMISTIC_actual_label", "OPTIMISTIC_response"),
    ]

    # Initialize lists for metrics
    metrics = []
    extracted_labels = {label: [] for _, label in label_pairs}

    # Process each label in batches
    batch_size = 10
    for actual_label, predicted_label in label_pairs:
        responses = data[predicted_label].tolist()
        actuals = data[actual_label].tolist()
        index_batches = chunk_list(list(range(len(responses))), batch_size)
        for batch_idx, batch_indices in enumerate(index_batches):
            response_batch = [responses[i] for i in batch_indices]
            extraction_prompt_func = get_prompt("subjectiveqa", PromptFormat.EXTRACTION)
            messages_batch = [
                [
                    {
                        "role": "user",
                        "content": extraction_prompt_func(resp, predicted_label),
                    }
                ]
                for resp in response_batch
            ]
            try:
                batch_responses = process_batch_with_retry(
                    args, messages_batch, batch_idx, len(index_batches)
                )
                for idx, (response, row_idx) in enumerate(
                    zip(batch_responses, batch_indices)
                ):
                    try:
                        if (
                            response is None
                            or not hasattr(response, "choices")
                            or not response.choices
                        ):
                            raise ValueError(f"Invalid API response: {response}")

                        llm_response = response.choices[0].message.content.strip()  # type: ignore
                        extracted_label = normalize_response(llm_response)

                        extracted_labels[predicted_label].append(
                            extracted_label if extracted_label is not None else -1
                        )

                    except Exception as e:
                        logger.error(
                            f"Error processing response for row {row_idx}: {e}"
                        )
                        extracted_labels[predicted_label].append(-1)

            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                extracted_labels[predicted_label].extend([-1] * len(batch_indices))
                continue
        # Compute metrics for the current label
        # assert len(actuals) == len(predicted_labels)
        actuals = [label if label in [0, 1, 2] else -1 for label in actuals]
        predicted_labels = extracted_labels[predicted_label]
        predicted_labels = [
            label if label in [0, 1, 2] else -1 for label in predicted_labels
        ]
        filtered_actuals = [
            actuals[i] for i in range(len(actuals)) if predicted_labels[i] != -1
        ]
        filtered_predictions = [
            predicted_labels[i]
            for i in range(len(predicted_labels))
            if predicted_labels[i] != -1
        ]
        if len(filtered_actuals) > 0 and len(filtered_predictions) > 0:
            precision = precision_score(
                filtered_actuals,
                filtered_predictions,
                average="weighted",
                zero_division=0,
            )
            recall = recall_score(
                filtered_actuals,
                filtered_predictions,
                average="weighted",
                zero_division=0,
            )
            f1 = f1_score(
                filtered_actuals,
                filtered_predictions,
                average="weighted",
                zero_division=0,
            )
            accuracy = accuracy_score(filtered_actuals, filtered_predictions)
        else:
            precision = recall = f1 = accuracy = (
                0.0  # If no valid labels, set metrics to zero
            )

        metrics.append(
            {
                "Label": predicted_label,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Accuracy": accuracy,
            }
        )

        logger.info(
            f"Metrics for {predicted_label}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}"
        )

    # Create metrics DataFrame
    results_df = pd.DataFrame(metrics)

    # Compute average metrics
    if len(metrics) > 0:
        average_precision = sum(result["Precision"] for result in metrics) / len(
            metrics
        )
        average_recall = sum(result["Recall"] for result in metrics) / len(metrics)
        average_f1 = sum(result["F1 Score"] for result in metrics) / len(metrics)
        average_accuracy = sum(result["Accuracy"] for result in metrics) / len(metrics)
    else:
        average_precision = average_recall = average_f1 = average_accuracy = 0.0
    logger.info(f"Average Precision: {average_precision:.4f}")
    logger.info(f"Average Recall: {average_recall:.4f}")
    logger.info(f"Average F1: {average_f1:.4f}")
    logger.info(f"Average Accuracy: {average_accuracy:.4f}")

    # Create DataFrame for aggregated statistics
    statistics_df = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
            "Average": [
                average_precision,
                average_recall,
                average_f1,
                average_accuracy,
            ],
        }
    )

    return results_df, statistics_df
