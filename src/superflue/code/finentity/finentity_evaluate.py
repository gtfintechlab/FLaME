import pandas as pd
import logging
from datetime import date
import json
import re
import ast
from litellm import batch_completion
from superflue.utils.logging_utils import setup_logger
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.code.extraction_prompts import finentity_extraction_prompt
from superflue.config import LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="finentity_evaluation",
    log_file=LOG_DIR / "finentity_evaluation.log",
    level=LOG_LEVEL,
)


def sanitize_json_string(json_str):
    """Sanitize JSON strings by fixing common formatting issues."""
    
    json_str = json_str.strip()
    json_str = json_str.replace(", }", "}").replace(", ]", "]") 
    json_str = json_str.replace("'", "\"") 
    json_str = json_str.replace("\\\"", "\"")  
    json_str = json_str.replace("“", "\"").replace("”", "\"") 
    json_str = re.sub(r'(?<!\\)"(s)', "'s", json_str)
    
    return json_str

def parse_json_content(content):
    """Parse JSON content with error handling."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(content)
        except (ValueError, SyntaxError) as e:
            logger.error(f"JSON decoding error: {e}")
            logger.error(f"Failed content: {content}")
            return []

def normalize_entities(entities):
    """Normalize entities for comparison."""
    return [
        {
            "value": entity["value"].strip().lower(),
            "tag": entity["tag"].strip().lower(),
            "label": entity["label"].strip().lower()
        }
        for entity in entities
        if "value" in entity and "tag" in entity and "label" in entity
    ]

def evaluate_entities(pred_entities, true_entities):
    """Evaluate entity extraction by comparing predicted and true entities."""
    normalized_pred = normalize_entities(pred_entities)
    normalized_true = normalize_entities(true_entities)

    matched = sum(1 for entity in normalized_pred if entity in normalized_true)
    unmatched_pred = len(normalized_pred) - matched
    unmatched_true = len(normalized_true) - matched

    precision = matched / (matched + unmatched_pred) if (matched + unmatched_pred) > 0 else 0
    recall = matched / (matched + unmatched_true) if (matched + unmatched_true) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = matched / len(normalized_true) if len(normalized_true) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

def finentity_evaluate(file_name, args):
    """Evaluate FinEntity dataset with batching."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    batch_size = args.batch_size
    indices = list(range(len(df)))
    index_batches = chunk_list(indices, batch_size)
    logger.info(f"Processing {len(df)} rows in {len(index_batches)} batches.")

    for batch_idx, batch_indices in enumerate(index_batches):
        llm_responses_batch = [df.at[i, "llm_responses"] for i in batch_indices]
        logger.info(f"Processing batch {batch_idx + 1}/{len(index_batches)} with {len(batch_indices)} rows.")
        messages_batch = [
            [{"role": "user", "content": finentity_extraction_prompt(response)}]
            for response in llm_responses_batch
        ]

        try:
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, len(index_batches))
            for idx, (response, row_idx) in enumerate(zip(batch_responses, batch_indices)):
                try:
                    if response is None or not hasattr(response, "choices") or not response.choices:
                        raise ValueError(f"Invalid API response: {response}")

                    llm_response = response.choices[0].message.content.strip()  # type: ignore
                    sanitized_label = sanitize_json_string(llm_response)
                    parsed_label = parse_json_content(sanitized_label)

                    df.at[row_idx, "extracted_labels"] = json.dumps(parsed_label)

                except Exception as e:
                    logger.error(f"Error processing response for row {row_idx}: {e}")
                    df.at[row_idx, "extracted_labels"] = json.dumps([])

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            for row_idx in batch_indices:
                df.at[row_idx, "extracted_labels"] = json.dumps([])

    # Evaluate extracted vs actual labels
    evaluation_results = []
    for index, row in df.iterrows():
        true_entities = parse_json_content(row["actual_labels"])
        pred_entities = parse_json_content(row["extracted_labels"])
        metrics = evaluate_entities(pred_entities, true_entities)
        evaluation_results.append(metrics)

    # Aggregate metrics
    aggregated_metrics = pd.DataFrame(evaluation_results).mean()
    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
        "Value": [
            aggregated_metrics["precision"],
            aggregated_metrics["recall"],
            aggregated_metrics["f1"],
            aggregated_metrics["accuracy"]
        ]
    })

    return df, metrics_df
