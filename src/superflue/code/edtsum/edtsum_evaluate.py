import pandas as pd
from datetime import date
from evaluate import load
import numpy as np
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL

bertscore = load("bertscore")

logger = setup_logger(
    name="edtsum_evaluation",
    log_file=LOG_DIR / "edtsum_evaluation.log",
    level=LOG_LEVEL,
)

def edtsum_evaluate(file_name, args):
    """Evaluate EDTSum temporal summaries and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    correct_summaries = df["actual_labels"].tolist()
    llm_responses = df["llm_responses"].tolist()

    logger.info("Computing BERTScore metrics...")
    bert_scores = bertscore.compute(
        predictions=llm_responses, 
        references=correct_summaries, 
        model_type="distilbert-base-uncased"
    )

    df["precision"] = bert_scores["precision"]  # type: ignore
    df["recall"] = bert_scores["recall"]  # type: ignore
    df["f1"] = bert_scores["f1"]  # type: ignore

    # Log aggregated metrics
    avg_precision = np.mean(bert_scores["precision"])  # type: ignore
    avg_recall = np.mean(bert_scores["recall"])  # type: ignore
    avg_f1 = np.mean(bert_scores["f1"])  # type: ignore

    logger.info(f"BERTScore Precision: {avg_precision:.4f}")
    logger.info(f"BERTScore Recall: {avg_recall:.4f}")
    logger.info(f"BERTScore F1: {avg_f1:.4f}")

    metrics_df = pd.DataFrame({
        "Precision": [avg_precision],
        "Recall": [avg_recall],
        "F1 Score": [avg_f1]
    })

    return df, metrics_df