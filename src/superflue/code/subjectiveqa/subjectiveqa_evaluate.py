from datetime import date
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL

# Setup logger
logger = setup_logger(
    name="subjectiveqa_evaluation",
    log_file=LOG_DIR / "subjectiveqa_evaluation.log",
    level=LOG_LEVEL,
)

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")
    
def subjectiveqa_evaluate(file_name, args):
    """Evaluate SubjectiveQA results and return evaluation and statistics DataFrames."""
    task = args.dataset.strip('“”"')
    
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    # Load the input CSV file
    data = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Define label pairs for evaluation
    label_pairs = [
        ("RELEVANT_actual_label", "RELEVANT"),
        ("SPECIFIC_actual_label", "SPECIFIC"),
        ("CAUTIOUS_actual_label", "CAUTIOUS"),
        ("ASSERTIVE_actual_label", "ASSERTIVE"),
        ("CLEAR_actual_label", "CLEAR"),
        ("OPTIMISTIC_actual_label", "OPTIMISTIC"),
    ]

    # Initialize lists for metrics
    precision_scores, recall_scores, f1_scores, accuracy_scores = [], [], [], []

    # Compute metrics for each label pair
    results_data = []
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    for actual, predicted in label_pairs:
        precision = precision_score(data[actual], data[predicted], average="weighted")
        recall = recall_score(data[actual], data[predicted], average="weighted")
        f1 = f1_score(data[actual], data[predicted], average="weighted")
        accuracy = accuracy_score(data[actual], data[predicted])

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

        results_data.append({
            "Label": predicted,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Accuracy": accuracy
        })

        logger.info(f"Metrics for {predicted}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}")
        results_df = pd.DataFrame(results_data)
        # Save progress after DataFrame creation
        save_progress(results_df, evaluation_results_path)

    # Compute average metrics
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

    logger.info(f"Average Precision: {average_precision:.4f}")
    logger.info(f"Average Recall: {average_recall:.4f}")
    logger.info(f"Average F1: {average_f1:.4f}")
    logger.info(f"Average Accuracy: {average_accuracy:.4f}")

    # Create DataFrame for aggregated statistics
    statistics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
        "Average": [average_precision, average_recall, average_f1, average_accuracy]
    })

    return results_df, statistics_df
