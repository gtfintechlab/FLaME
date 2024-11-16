import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together
from together import Together
from superflue.config import RESULTS_DIR, ROOT_DIR
from superflue.utils.logging_utils import setup_logger
import os
import json
import ast

client = Together()
logger = setup_logger(
    name="finentity_evaluate",
    log_file=Path("logs/finentity_evaluate.log"),
    level=logging.INFO,
)

INPUT_FILE_PATH = os.path.join(RESULTS_DIR, "finentity", "finentity_meta-llama", "Meta-Llama-3.1-8B-Instruct-Turbo_02_10_2024.csv")

# Define prompt for formatting JSON output
def finentity_prompt(model_response: str):
    prompt = f"""Reformat the following extracted entity list into a structured JSON array.
                Use the exact format below, ensuring each entity has "start", "end", "value", "tag", and "label".
                Return only the JSON list, with no additional text.

                Original output:
                {model_response}

                Example format:
                [
                {{"start": 0, "end": 15, "value": "EntityName", "tag": "NEUTRAL", "label": "NEUTRAL"}},
                {{"start": 16, "end": 30, "value": "EntityName2", "tag": "POSITIVE", "label": "POSITIVE"}}
                ]

                Please ensure the format is valid JSON with all required fields."""
    return prompt

# Sanitize JSON string to fix common issues before parsing
def sanitize_json_string(json_str):
    json_str = json_str.strip()
    json_str = json_str.replace(", }", "}").replace(", ]", "]")  # Remove trailing commas
    json_str = json_str.replace("'", "\"")  # Replace single quotes with double quotes
    return json_str

# Enhanced JSON parsing with error handling
def parse_json_content(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(content)
        except (ValueError, SyntaxError) as e:
            logger.error(f"JSON decoding error: {e}")
            logger.error(f"Failed content: {content}")
            return []

# Helper function to save progress to CSV
def save_progress(df, path):
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

# Evaluation function for matching extracted and actual entities
def evaluate_entities(pred_entities, true_entities):
    matched, unmatched_pred, unmatched_true = [], [], []

    for true_entity in true_entities:
        match_found = False
        for pred_entity in pred_entities:
            iou = compute_iou(true_entity, pred_entity)
            if iou > 0.5 and true_entity["tag"] == pred_entity["tag"]:
                matched.append(pred_entity)
                match_found = True
                break
        if not match_found:
            unmatched_true.append(true_entity)

    unmatched_pred = [e for e in pred_entities if e not in matched]

    precision = len(matched) / (len(matched) + len(unmatched_pred)) if len(matched) + len(unmatched_pred) > 0 else 0
    recall = len(matched) / (len(matched) + len(unmatched_true)) if len(matched) + len(unmatched_true) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = len(matched) / len(true_entities) if len(true_entities) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

# IoU function for matching entity position
def compute_iou(entity_a, entity_b):
    start_a, end_a = entity_a["start"], entity_a["end"]
    start_b, end_b = entity_b["start"], entity_b["end"]
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return intersection / union if union > 0 else 0

# Function to extract and evaluate entities from responses
def extract_and_evaluate_entities(args):
    results_file = INPUT_FILE_PATH
    df = pd.read_csv(results_file)
    sentences = df['llm_responses'].tolist()

    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / args.task
        / f"evaluation_{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    if 'extracted_labels' not in df.columns:
        df['extracted_labels'] = None

    for i, sentence in enumerate(sentences):
        if pd.notna(df.at[i, 'extracted_labels']):
            continue

        try:
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a JSON formatter for entity extraction outputs."},
                    {"role": "user", "content": finentity_prompt(sentence)},
                ],
                max_tokens=512,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_label = model_response.choices[0].message.content.strip()
            extracted_label = sanitize_json_string(extracted_label)

            logger.info(f"Raw model response for row {i}: {extracted_label}")

            parsed_label = parse_json_content(extracted_label)
            df.at[i, 'extracted_labels'] = json.dumps(parsed_label)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            df.at[i, 'extracted_labels'] = json.dumps([])

    evaluation_results = []
    for index, row in df.iterrows():
        true_entities = parse_json_content(row['actual_labels'])
        pred_entities = parse_json_content(row['extracted_labels'])
        metrics = evaluate_entities(pred_entities, true_entities)
        evaluation_results.append(metrics)

    aggregated_metrics = pd.DataFrame(evaluation_results).mean()
    logger.info(f"Aggregated Precision: {aggregated_metrics['precision']:.4f}")
    logger.info(f"Aggregated Recall: {aggregated_metrics['recall']:.4f}")
    logger.info(f"Aggregated F1 Score: {aggregated_metrics['f1']:.4f}")
    logger.info(f"Aggregated Accuracy: {aggregated_metrics['accuracy']:.4f}")

    eval_df = pd.DataFrame({
        "Precision": [aggregated_metrics['precision']],
        "Recall": [aggregated_metrics['recall']],
        "F1 Score": [aggregated_metrics['f1']],
        "Accuracy": [aggregated_metrics['accuracy']]
    })
    eval_df.to_csv(Path(f"{str(evaluation_results_path)[:-4]}_statistics.csv"), index=False)
    df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Evaluation completed. Results saved to {evaluation_results_path}")

    return df, eval_df

tokens_map = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ["["],
}

def tokens(model_name):
    return tokens_map.get(model_name, [])

if __name__ == "__main__":
    class Args:
        task = "finentity"
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        date = date.today().strftime('%d_%m_%Y')
        max_tokens = 256
        temperature = 0.0
        top_p = 0.9
        repetition_penalty = 1.0

    args = Args()
    extract_and_evaluate_entities(args)
