import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
from datetime import date

# Configure logging
logger = setup_logger(
    name="finentity_evaluation",
    log_file=LOG_DIR / "finentity_evaluation.log",
    level=LOG_LEVEL,
)

def adjust_tags(row):
    actual = row["actual_tags"]
    predicted = row["predicted_tags"]
    if len(predicted) > len(actual):
        return predicted[:len(actual)]
    elif len(predicted) < len(actual):
        return None  # Mark for exclusion
    else:
        return predicted

def casual_detection_evaluate(file_name, args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    df["actual_tags"] = df["actual_tags"].apply(eval)  # Convert string to list
    df["predicted_tags"] = df["predicted_tags"].apply(eval)  # Convert string to list
    df["adjusted_predicted_tags"] = df.apply(adjust_tags, axis=1) # type: ignore

    # Exclude rows with mismatched lengths after adjustment
    df["length_match"] = df["adjusted_predicted_tags"].notnull()

    df["row_accuracy"] = df.apply(
        lambda row: accuracy_score(row["actual_tags"], row["adjusted_predicted_tags"]) # type: ignore
        if row["length_match"] else 0.0, 
        axis=1
    ) 

    valid_rows = df[df["length_match"]]

    flat_actual = [tag for tags in valid_rows["actual_tags"] for tag in tags]
    flat_predicted = [tag for tags in valid_rows["adjusted_predicted_tags"] for tag in tags]

    labels = ["B-CAUSE", "I-CAUSE", "B-EFFECT", "I-EFFECT", "O"]
    print("Token Classification Report:")
    print(classification_report(flat_actual, flat_predicted, labels=labels))

    accuracy = accuracy_score(flat_actual, flat_predicted)
    print(f"Overall Token-Level Accuracy: {accuracy:.4f}")

    accuracy = accuracy_score(flat_actual, flat_predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(flat_actual, flat_predicted, average="weighted")

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    df.to_csv(evaluation_results_path, index=False)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
    })

    # Save metrics DataFrame
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df