import pandas as pd
import logging
from datetime import date
from pathlib import Path
from superflue.code.tokens import tokens
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list
from superflue.code.extraction_prompts import fiqa_1_extraction_prompt
import warnings
import argparse
import re
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
from superflue.utils.logging_utils import setup_logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import litellm
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

# Setup logger
logger = setup_logger(
    name="convfinqa_evaluation",
    log_file=LOG_DIR / "convfinqa_evaluation.log",
    level=LOG_LEVEL,
)



def extract_numerical_value(text):
    match = re.search(r"(-?\d+\.\d+)", text)  # Adjusted to capture decimal values
    return float(match.group(0)) if match else None


def fiqa_task1_evaluate(file_name, args):

    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Output path for evaluation results
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    extraction_response = []
    extraction_model_response = []
    regex_extraction = []

    all_responses = df["llm_responses"].tolist()
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": fiqa_1_extraction_prompt(llm_response)}]
            for llm_response in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(batch)):
                extraction_response.append(None)
                regex_extraction.append(None)
                extraction_model_response.append(str(e))

        for response in batch_responses:
            extraction_model_response.append(response)
            try:
                response_text = response.choices[0].message.content  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_text = None
            extraction_response.append(response_text)
            numerical_value = extract_numerical_value(response_text)
            regex_extraction.append(numerical_value)

    df['extraction_model_response'] = extraction_model_response
    df['extraction_response'] = extraction_response
    df['regex_extraction'] = regex_extraction    

    correct_labels = df['actual_sentiment'].tolist()    

    # add else statement that just puts None if none
    correct_labels = [str(label) if label is not None else "Error" for label in correct_labels]
    regex_extraction = [str(label) if label is not None else "Error" for label in regex_extraction]

    # Calculate metrics
    accuracy = accuracy_score(correct_labels, regex_extraction) 
    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     correct_labels, regex_extraction
    # )

    # # Log metrics
    # logger.info(f"Accuracy: {accuracy:.4f}")
    # logger.info(f"Precision: {precision:.4f}")
    # logger.info(f"Recall: {recall:.4f}")
    # logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy"],
        "Value": [accuracy],
    })
    # metrics_df = pd.DataFrame({
    #     "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    #     "Value": [accuracy, precision, recall, f1],
    # })

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    df.to_csv(evaluation_results_path, index=False)

    # Save metrics DataFrame
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
