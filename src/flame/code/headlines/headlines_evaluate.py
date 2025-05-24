import json
import pandas as pd
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm
import ast
from flame.code.prompts.registry import get_prompt, PromptFormat


# Configure logging
logger = setup_logger(
    name="headlines_evaluation",
    log_file=LOG_DIR / "headlines_evaluation.log",
    level=LOG_LEVEL,
)

label_mapping = {
    "Price_or_Not": {"0": 0, "1": 1},
    "Direction_Up": {"0": 0, "1": 1},
    "Direction_Down": {"0": 0, "1": 1},
    "Direction_Constant": {"0": 0, "1": 1},
    "Past_Price": {"0": 0, "1": 1},
    "Future_Price": {"0": 0, "1": 1},
    "Past_News": {"0": 0, "1": 1},
}


def preprocess_llm_response(raw_response: str):
    """Preprocess the raw LLM response to extract JSON content."""
    try:
        # Remove Markdown-style code fencing
        if raw_response.startswith("```json"):
            raw_response = raw_response.split("```json")[1].split("```")[0].strip()
        elif raw_response.startswith("```"):
            raw_response = raw_response.split("```")[1].split("```")[0].strip()
        return raw_response
    except Exception as e:
        logger.error(f"Error preprocessing LLM response: {e}")
        return None


def map_label_to_number(label: str, category: str):
    """Map extracted labels to numeric values."""
    return label_mapping[category].get(label.strip(), -1)


def save_progress(df, path):
    """Save progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")


def headlines_evaluate(file_name, args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Initialize extracted labels if not present
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    actual_labels = df["actual_labels"].tolist()
    actual_predictions = [ast.literal_eval(labels) for labels in actual_labels]
    extracted_labels = []

    all_responses = df["llm_responses"].tolist()
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch_content in enumerate(pbar):
        extraction_prompt_func = get_prompt("headlines", PromptFormat.EXTRACTION)
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(response)}]
            for response in batch_content
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(batch_content)):
                extracted_labels.append([-1] * 7)

        for response in batch_responses:
            try:
                raw_response = response.choices[0].message.content.strip()
                preprocessed_response = preprocess_llm_response(raw_response)
                if not preprocessed_response:
                    raise ValueError(
                        f"Preprocessing failed for response: {raw_response}"
                    )
                extracted_label_json = json.loads(preprocessed_response)
            except Exception as e:
                logger.error(f"Error extracting response: {e}")
                extracted_labels.append([-1] * 7)
                continue

            mapped_labels = [
                map_label_to_number(
                    str(extracted_label_json.get("Price_or_Not", "")), "Price_or_Not"
                ),
                map_label_to_number(
                    str(extracted_label_json.get("Direction_Up", "")), "Direction_Up"
                ),
                map_label_to_number(
                    str(extracted_label_json.get("Direction_Down", "")),
                    "Direction_Down",
                ),
                map_label_to_number(
                    str(extracted_label_json.get("Direction_Constant", "")),
                    "Direction_Constant",
                ),
                map_label_to_number(
                    str(extracted_label_json.get("Past_Price", "")), "Past_Price"
                ),
                map_label_to_number(
                    str(extracted_label_json.get("Future_Price", "")), "Future_Price"
                ),
                map_label_to_number(
                    str(extracted_label_json.get("Past_News", "")), "Past_News"
                ),
            ]
            extracted_labels.append(mapped_labels)

    # Metrics

    df["extracted_labels"] = extracted_labels

    accuracies = []

    for extracted, actual in zip(extracted_labels, actual_predictions):
        acc = 0
        for e, a in zip(extracted, actual):
            if e == a:
                acc += 1
        accuracies.append(acc / len(actual))

    accuracy = sum(accuracies) / len(accuracies)

    metrics_df = pd.DataFrame({"Metric": ["Accuracy"], "Value": [accuracy]})

    logger.info(f"Accuracy: {accuracy:.4f}")

    return df, metrics_df
