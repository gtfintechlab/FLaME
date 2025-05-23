import pandas as pd
from evaluate import load
import numpy as np
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_DIR, LOG_LEVEL

# Configure logging first so we can use it in get_bertscore
logger = setup_logger(
    name="ectsum_evaluation",
    log_file=LOG_DIR / "ectsum_evaluation.log",
    level=LOG_LEVEL,
)

# Load BERTScore evaluation metric lazily
_bertscore = None
_bertscore_error = None


def get_bertscore():
    global _bertscore, _bertscore_error

    # If we already tried and failed, raise the error
    if _bertscore_error is not None:
        raise _bertscore_error

    # If not loaded yet, try to load it
    if _bertscore is None:
        try:
            logger.info("Loading BERTScore metric...")
            _bertscore = load("bertscore")
            logger.info("BERTScore loaded successfully")
        except Exception as e:
            _bertscore_error = RuntimeError(
                f"Failed to load BERTScore metric. Make sure 'bert-score' is installed: {str(e)}"
            )
            raise _bertscore_error

    return _bertscore


def summarization_prompt(input_text: str):
    """Generate a summarization prompt for ECT data."""
    prompt = f'''Generate a financial summary in about 50 words in line-by-line format based on the following input. The summary should include key financial information such as earnings per share, revenue, and other significant figures.
                It should contain only lower case letters and numbers (including decimals). Do not include any special characters other than \n, % or $.
                Here is the input to analyze:
                "{input_text}"'''
    return prompt


def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")


def ectsum_evaluate(file_name, args):
    """Evaluate ECTSum summaries and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Note: Path definition removed - evaluate.py handles saving

    # Extract references and predictions
    correct_summaries = df["actual_labels"].tolist()
    llm_responses = df["llm_responses"].tolist()

    # Compute BERTScore
    logger.info("Computing BERTScore metrics...")
    bert_scores = get_bertscore().compute(
        predictions=llm_responses,
        references=correct_summaries,
        model_type="distilbert-base-uncased",
    )

    # Add BERTScore metrics to DataFrame
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

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1 Score"],
            "Value": [avg_precision, avg_recall, avg_f1],
        }
    )

    # Note: Progress and metrics saving removed - evaluate.py handles saving

    return df, metrics_df
