import time
from datetime import date
import os
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import together
from together import Together

from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, ROOT_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="subjectiveqa_evaluation",
    log_file=LOG_DIR / "subjectiveqa_evaluation.log",
    level=LOG_LEVEL,
)

INPUT_FILE_PATH = os.path.join(RESULTS_DIR, "subjectiveqa", "subjectiveqa_meta-llama", "Meta-Llama-3.1-8B-Instruct-Turbo_02_10_2024.csv")


def evaluate_subjectiveqa(INPUT_FILE_PATH):
    
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / 'subjectiveqa'
        / f"evaluation_{'subjectiveqa'}_{'meta-llama-3.1-8b'}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    

    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = pd.read_csv(INPUT_FILE_PATH)
    
    precision_scores, recall_scores, f1_scores, accuracy_scores = [], [], [], []

    label_pairs = [
        ("RELEVANT_actual_label", "RELEVANT"),
        ("SPECIFIC_actual_label", "SPECIFIC"),
        ("CAUTIOUS_actual_label", "CAUTIOUS"),
        ("ASSERTIVE_actual_label", "ASSERTIVE"),
        ("CLEAR_actual_label", "CLEAR"),
        ("OPTIMISTIC_actual_label", "OPTIMISTIC")
    ]

    for actual, predicted in label_pairs:
        precision = precision_score(data[actual], data[predicted], average="weighted")
        recall = recall_score(data[actual], data[predicted], average="weighted")
        f1 = f1_score(data[actual], data[predicted], average="weighted")
        accuracy = accuracy_score(data[actual], data[predicted])

        # Append scores to lists
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    
    for actual, predicted in label_pairs:
        logger.info(f"Precision for {predicted}: {precision}")
        logger.info(f"Recall for {predicted}: {recall}")
        logger.info(f"F1 for {predicted}: {f1}")
        logger.info(f"Accuracy for {predicted}: {accuracy}")
        
        
    logger.info(f"Average Precision: {average_precision}")
    logger.info(f"Average Recall: {average_recall}")
    logger.info(f"Average F1: {average_f1}")
    logger.info(f"Average Accuracy: {average_accuracy}")
    
    df = pd.DataFrame({
        "Average Precision": [average_precision],
        "Average Recall": [average_recall],
        "Average F1": [average_f1],
        "Average Accuracy": [average_accuracy],
    })
    
    for i, (actual, predicted) in enumerate(label_pairs):
        df[f"{predicted}_Precision"] = precision_scores[i]
        df[f"{predicted}_Recall"] = recall_scores[i]
        df[f"{predicted}_F1"] = f1_scores[i]
        df[f"{predicted}_Accuracy"] = accuracy_scores[i]
    
    df.to_csv(evaluation_results_path, index=False)
    
    
if __name__ == "__main__":
    evaluate_subjectiveqa(INPUT_FILE_PATH)