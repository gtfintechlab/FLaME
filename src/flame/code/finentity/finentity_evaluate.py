import ast
import json
import re

import pandas as pd
from tqdm import tqdm

from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger(
    name="finentity_evaluation",
    log_file=LOG_DIR / "finentity_evaluation.log",
    level=LOG_LEVEL,
)


def finentity_prompt(model_response: str):
    """Generate a prompt to reformat extracted entity lists into structured JSON."""
    prompt = f"""Reformat the following extracted entity list into a structured JSON array.
                Use the exact format below, ensuring each entity has 'value', 'tag', and 'label'.
                Return only the JSON list, with no additional text.

                Original output:
                {model_response}

                Example format:
                [
                {{'value': 'EntityName', 'tag': 'NEUTRAL', 'label': 'NEUTRAL'}},
                {{'value': 'EntityName2', 'tag': 'POSITIVE', 'label': 'POSITIVE'}}
                ]

                Please ensure the format is valid JSON with all required fields. Make sure it does not throw a JSON decoding error."""
    return prompt


def sanitize_json_string(json_str):
    """Sanitize JSON strings by fixing common formatting issues."""

    json_str = json_str.strip()

    # Remove markdown code blocks if present
    if json_str.startswith("```"):
        # Find the end of the opening code block marker
        first_newline = json_str.find("\n")
        if first_newline != -1:
            json_str = json_str[first_newline + 1 :]
        # Remove the closing code block marker
        if json_str.endswith("```"):
            json_str = json_str[:-3].rstrip()
    json_str = json_str.replace(", }", "}").replace(
        ", ]", "]"
    )  # Remove trailing commas
    json_str = json_str.replace("'", '"')  # Ensure JSON uses double quotes
    json_str = json_str.replace('\\"', '"')  # Fix double-escaped quotes
    json_str = json_str.replace("“", '"').replace("”", '"')  # Handle curly quotes
    json_str = re.sub(
        r'(?<!\\)"(s)', "'s", json_str
    )  # Fix possessive errors (e.g., Lowe"s → Lowe's)

    return json_str


def parse_json_content(content):
    """Parse JSON content with error handling."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Try to fix common issues
        try:
            # Handle cases like 'stap" by finding and fixing mismatched quotes
            fixed_content = content

            # Count quotes to detect mismatches
            double_quotes = fixed_content.count('"')
            single_quotes = fixed_content.count("'")

            # If odd number of quotes, likely a mismatch
            if double_quotes % 2 != 0 or single_quotes % 2 != 0:
                # Try to parse line by line and fix
                lines = fixed_content.split("\n")
                for i, line in enumerate(lines):
                    if '"value":' in line and line.count('"') % 2 != 0:
                        # Find the problematic quote and try to fix it
                        if line.strip().endswith('",'):
                            lines[i] = line[:-2] + '"},'
                        elif "'" in line:
                            # Replace single quotes in values
                            lines[i] = line.replace("'", '"')
                fixed_content = "\n".join(lines)

            return json.loads(fixed_content)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(content)
            except (ValueError, SyntaxError):
                logger.error(f"JSON decoding error: {e}")
                logger.error(f"Failed content: {content}")
                return []


def normalize_entities(entities):
    """Normalize entities for comparison."""
    return [
        {
            "value": entity["value"].strip().lower(),
            "tag": entity["tag"].strip().lower(),
            "label": entity["label"].strip().lower(),
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

    precision = (
        matched / (matched + unmatched_pred) if (matched + unmatched_pred) > 0 else 0
    )
    recall = (
        matched / (matched + unmatched_true) if (matched + unmatched_true) > 0 else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = matched / len(normalized_true) if len(normalized_true) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def finentity_evaluate(file_name, args):
    """Evaluate FinEntity dataset with batching."""
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "finentity"
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    # Batching
    batch_size = args.batch_size
    indices = list(range(len(df)))
    index_batches = chunk_list(indices, batch_size)
    logger.info(f"Processing {len(df)} rows in {len(index_batches)} batches.")

    # Extract labels
    pbar = tqdm(index_batches, desc="Processing batches")
    for batch_idx, batch_indices in enumerate(pbar):
        llm_responses_batch = [df.at[i, "llm_responses"] for i in batch_indices]
        pbar.set_description(f"Batch {batch_idx + 1}/{len(index_batches)}")
        messages_batch = [
            [{"role": "user", "content": finentity_prompt(response)}]
            for response in llm_responses_batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, len(index_batches)
            )
            for idx, (response, row_idx) in enumerate(
                zip(batch_responses, batch_indices)
            ):
                try:
                    if (
                        response is None
                        or not hasattr(response, "choices")
                        or not response.choices
                    ):
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
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
            "Value": [
                aggregated_metrics["precision"],
                aggregated_metrics["recall"],
                aggregated_metrics["f1"],
                aggregated_metrics["accuracy"],
            ],
        }
    )

    return df, metrics_df
