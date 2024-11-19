import pandas as pd
import logging
from datetime import date
from pathlib import Path
import json
import ast
from litellm import completion
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL

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

                Please ensure the format is valid JSON with all required fields."""
    return prompt

def sanitize_json_string(json_str):
    """Sanitize JSON strings by fixing common formatting issues."""
    json_str = json_str.strip()
    json_str = json_str.replace(", }", "}").replace(", ]", "]")  # Remove trailing commas
    json_str = json_str.replace("'", "\"")  # Replace single quotes with double quotes
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

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def finentity_evaluate(file_name, args):
    """Evaluate FinEntity dataset and return results and metrics DataFrames."""
    task = "finentity"
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    # Extract labels
    for i, sentence in enumerate(df["llm_responses"]):
        if pd.notna(df.at[i, "extracted_labels"]):
            continue

        try:
            # Generate prompt and get response
            response = completion(
                model=args.model,
                messages=[
                    {"role": "user", "content": finentity_prompt(sentence)},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_label = response.choices[0].message.content.strip() # type: ignore
            sanitized_label = sanitize_json_string(extracted_label)
            parsed_label = parse_json_content(sanitized_label)

            df.at[i, "extracted_labels"] = json.dumps(parsed_label)
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            df.at[i, "extracted_labels"] = json.dumps([])

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
        "Precision": [aggregated_metrics["precision"]],
        "Recall": [aggregated_metrics["recall"]],
        "F1 Score": [aggregated_metrics["f1"]],
        "Accuracy": [aggregated_metrics["accuracy"]]
    })

    # Save metrics
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
