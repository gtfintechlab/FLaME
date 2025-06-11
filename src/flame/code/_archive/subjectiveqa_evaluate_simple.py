import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from flame.utils.logging_utils import get_component_logger

# Setup logger
logger = get_component_logger("evaluation", "subjectiveqa")


def subjectiveqa_evaluate(file_name, args):
    """Evaluate SubjectiveQA results directly without extraction."""
    # support legacy args.dataset for tests, prefer args.task
    task = (
        getattr(args, "task", None) or getattr(args, "dataset", None) or "subjectiveqa"
    )
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

    # Process each label directly (no extraction needed)
    for actual_label, predicted_label in label_pairs:
        actuals = data[actual_label].tolist()
        predictions = data[predicted_label].tolist()

        # Filter out any invalid values
        filtered_actuals = []
        filtered_predictions = []
        for a, p in zip(actuals, predictions):
            if a in [0, 1, 2] and p in [0, 1, 2]:
                filtered_actuals.append(a)
                filtered_predictions.append(p)

        if len(filtered_actuals) > 0:
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
            precision = recall = f1 = accuracy = 0.0

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
    metrics_df = pd.DataFrame(metrics)

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

    # Add average metrics to the metrics DataFrame
    average_row = {
        "Label": "AVERAGE",
        "Precision": average_precision,
        "Recall": average_recall,
        "F1 Score": average_f1,
        "Accuracy": average_accuracy,
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([average_row])], ignore_index=True)

    # Add extracted_labels columns to match expected format
    for _, predicted_label in label_pairs:
        data[f"{predicted_label}_extracted"] = data[predicted_label]

    # Return both detailed metrics and statistics
    return data, metrics_df
