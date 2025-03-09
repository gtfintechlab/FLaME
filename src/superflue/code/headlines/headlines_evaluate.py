import json
import pandas as pd
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list
from superflue.code.extraction_prompts import headlines_extraction_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
from tqdm import tqdm
import ast

# Configure logging
logger = setup_logger(
    name="headlines_evaluation",
    log_file=LOG_DIR / "headlines_evaluation.log",
    level=LOG_LEVEL,
)

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

def headlines_evaluate(file_name, args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    actual_labels = df['actual_labels'].tolist()
    actual_predictions = [ast.literal_eval(labels) for labels in actual_labels]
    extracted_labels = []

    all_responses = df["llm_responses"].tolist()
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch_content in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": headlines_extraction_prompt(response)}]
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
            continue

        for response in batch_responses:
            try: 
                raw_response = response.choices[0].message.content.strip()
                preprocessed_response = preprocess_llm_response(raw_response)
                if not preprocessed_response:
                    raise ValueError(f"Preprocessing failed for response: {raw_response}")
                extracted_label_json = json.loads(preprocessed_response)
            except Exception as e:
                logger.error(f"Error extracting response: {e}")
                extracted_labels.append([-1] * 7)
                continue
                
            mapped_labels = [
                int(extracted_label_json.get("Price_or_Not", "")),
                int(extracted_label_json.get("Direction_Up", "")),
                int(extracted_label_json.get("Direction_Down", "")),
                int(extracted_label_json.get("Direction_Constant", "")),
                int(extracted_label_json.get("Past_Price", "")),
                int(extracted_label_json.get("Future_Price", "")),
                int(extracted_label_json.get("Past_News", ""))
            ]
            extracted_labels.append(mapped_labels)
        
        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

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

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy"],
        "Value": [accuracy]
    })

    logger.info(f"Accuracy: {accuracy:.4f}")

    success_rate = df["extracted_labels"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
