import os
import pandas as pd
from datetime import date
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import together
from together import Together
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, ROOT_DIR, LOG_DIR, LOG_LEVEL

# Set up logging
logger = setup_logger(
    name="causal_classification_evaluation",
    log_file=LOG_DIR / "causal_classification_evaluation.log",
    level=LOG_LEVEL,
)

# Define input file path
INPUT_FILE_PATH = os.path.join(RESULTS_DIR, "causal_classification", "causal_classification_meta-llama", "Meta-Llama-3.1-8B-Instruct-Turbo_02_10_2024.csv")

# Function to evaluate causal classification
def evaluate_causal_classification(INPUT_FILE_PATH):
    # Define evaluation results path
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / "causal_classification"
        / f"evaluation_causal_classification_meta-llama-3.1-8b_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    
    # Ensure directory exists
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = pd.read_csv(INPUT_FILE_PATH)
    
    # Ensure required columns are present
    if 'actual_labels' not in data.columns or 'llm_responses' not in data.columns:
        logger.error("The input CSV must contain 'actual_labels' and 'llm_responses' columns.")
        return

    # Lists to store metrics
    precision_scores, recall_scores, f1_scores, accuracy_scores = [], [], [], []

    # Calculate metrics for 'llm_responses' predictions against 'actual_labels'
    precision = precision_score(data['actual_labels'], data['llm_responses'], average="weighted")
    recall = recall_score(data['actual_labels'], data['llm_responses'], average="weighted")
    f1 = f1_score(data['actual_labels'], data['llm_responses'], average="weighted")
    accuracy = accuracy_score(data['actual_labels'], data['llm_responses'])

    # Append scores to lists
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    accuracy_scores.append(accuracy)

    # Log individual metrics
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Calculate and log average metrics
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    
    logger.info(f"Average Precision: {average_precision}")
    logger.info(f"Average Recall: {average_recall}")
    logger.info(f"Average F1: {average_f1}")
    logger.info(f"Average Accuracy: {average_accuracy}")
    
    # Save results to DataFrame and CSV
    results_df = pd.DataFrame({
        "Average Precision": [average_precision],
        "Average Recall": [average_recall],
        "Average F1": [average_f1],
        "Average Accuracy": [average_accuracy],
    })
    
    results_df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Evaluation results saved to {evaluation_results_path}")

# Run the evaluation
if __name__ == "__main__":
    evaluate_causal_classification(INPUT_FILE_PATH)
