import pandas as pd
from flame.utils.batch_utils import process_batch_with_retry, chunk_list
from flame.code.extraction_prompts import fiqa_1_extraction_prompt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.logging_utils import setup_logger
from tqdm import tqdm
import numpy as np

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
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

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
            continue

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

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df["extraction_model_response"] = extraction_model_response
    df["extraction_response"] = extraction_response
    df["regex_extraction"] = regex_extraction

    correct_labels = df["actual_sentiment"].tolist()
    correct_labels = [extract_numerical_value(label) for label in correct_labels]

    count_missing = 0

    for i in range(len(correct_labels)):
        if np.isnan(regex_extraction[i]):
            count_missing += 1
            if correct_labels[i] >= 0:  # type: ignore
                regex_extraction[i] = correct_labels[i] - 2  # type: ignore
            else:
                regex_extraction[i] = correct_labels[i] + 2  # type: ignore

    mse = mean_squared_error(correct_labels, regex_extraction)
    mae = mean_absolute_error(correct_labels, regex_extraction)
    r2 = r2_score(correct_labels, regex_extraction)
    answer_coverage = (len(correct_labels) - count_missing) / len(correct_labels)

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Mean Squared Error",
                "Mean Absolute Error",
                "R2 Score",
                "Answer Coverage",
            ],
            "Value": [mse, mae, r2, answer_coverage],
        }
    )

    success_rate = df["regex_extraction"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
