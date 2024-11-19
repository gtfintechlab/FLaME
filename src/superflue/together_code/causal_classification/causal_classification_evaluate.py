import os
import pandas as pd
from datetime import date
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL

# Set up logging
logger = setup_logger(
    name="causal_classification_evaluation",
    log_file=LOG_DIR / "causal_classification_evaluation.log",
    level=LOG_LEVEL,
)

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def causal_classification_evaluate(file_name, args):
    """Evaluate causal classification results and return results and metrics DataFrames."""
    task = "causal_classification"
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Ensure required columns are present
    if 'actual_labels' not in df.columns or 'llm_responses' not in df.columns:
        logger.error("The input CSV must contain 'actual_labels' and 'llm_responses' columns.")
        raise ValueError("Missing required columns in the input file.")

    # Define evaluation results path
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    logger.info("Calculating evaluation metrics...")
    precision = precision_score(df['actual_labels'], df['llm_responses'], average="weighted", zero_division=0)
    recall = recall_score(df['actual_labels'], df['llm_responses'], average="weighted", zero_division=0)
    f1 = f1_score(df['actual_labels'], df['llm_responses'], average="weighted", zero_division=0)
    accuracy = accuracy_score(df['actual_labels'], df['llm_responses'])

    # Add metrics to DataFrame
    df["precision"] = precision
    df["recall"] = recall
    df["f1"] = f1
    df["accuracy"] = accuracy

    # Log metrics
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Save progress
    save_progress(df, evaluation_results_path)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "Accuracy": [accuracy],
    })

    # Save metrics
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
