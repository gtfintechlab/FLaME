import pandas as pd
import logging
import json
import ast
from litellm import completion
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.path_utils import get_evaluation_save_path
from superflue.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm

# Configure logging
logger = setup_logger(
    name="finentity_evaluation",
    log_file=LOG_DIR / "finentity_evaluation.log",
    level=LOG_LEVEL,
)

def finentity_prompt(model_response: str) -> str:
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

def sanitize_json_string(json_str: str) -> str:
    """Clean and format JSON string for parsing."""
    try:
        # Remove any text before the first '[' and after the last ']'
        start_idx = json_str.find('[')
        end_idx = json_str.rfind(']') + 1
        if start_idx == -1 or end_idx == 0:
            return "[]"
        return json_str[start_idx:end_idx]
    except Exception as e:
        logger.error(f"Error sanitizing JSON string: {e}")
        return "[]"

def parse_json_content(content: str) -> list:
    """Parse JSON content into a list of dictionaries."""
    try:
        if pd.isna(content):
            return []
        if isinstance(content, str):
            content = ast.literal_eval(content)
        return content if isinstance(content, list) else []
    except Exception as e:
        logger.error(f"Error parsing JSON content: {e}")
        return []

def evaluate_entities(pred_entities: list, true_entities: list) -> dict:
    """Evaluate predicted entities against true entities."""
    try:
        # Convert entities to sets of tuples for comparison
        true_set = {(e['value'], e['tag'], e['label']) for e in true_entities}
        pred_set = {(e['value'], e['tag'], e['label']) for e in pred_entities}

        # Calculate metrics
        true_positives = len(true_set.intersection(pred_set))
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)

        # Calculate precision, recall, F1, and accuracy
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }
    except Exception as e:
        logger.error(f"Error evaluating entities: {e}")
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "accuracy": 0
        }

def finentity_evaluate(file_name: str, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate FinEntity dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('"""')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths using consistent utility
    evaluation_results_path = get_evaluation_save_path(args.dataset, args.model)
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    # Extract labels
    for i, sentence in tqdm(enumerate(df["llm_responses"]), desc="Processing responses", total=len(df["llm_responses"])):
        if pd.notna(df.at[i, "extracted_labels"]):
            continue

        try:
            # Generate prompt and get response
            response = completion(
                model=args.model,
                messages=[{"role": "user", "content": finentity_prompt(sentence)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_label = response.choices[0].message.content.strip() # type: ignore
            sanitized_label = sanitize_json_string(extracted_label)
            parsed_label = parse_json_content(sanitized_label)

            # Update DataFrame
            df.at[i, "extracted_labels"] = json.dumps(parsed_label)
            df.to_csv(evaluation_results_path, index=False)

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

    # Create metrics DataFrame with consistent format
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [
            aggregated_metrics["accuracy"],
            aggregated_metrics["precision"],
            aggregated_metrics["recall"],
            aggregated_metrics["f1"]
        ],
    })

    # Save metrics using consistent naming
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
