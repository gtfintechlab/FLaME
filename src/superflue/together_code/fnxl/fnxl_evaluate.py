import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together
from together import Together
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from superflue.config import RESULTS_DIR, ROOT_DIR
from superflue.utils.logging_utils import setup_logger
import os
import json

client = Together()
logger = setup_logger(
    name="fnxl_evaluate",
    log_file=Path("logs/fnxl_evaluate.log"),
    level=logging.INFO,
)

INPUT_FILE_PATH = os.path.join(RESULTS_DIR, "fnxl", "fnxl_meta-llama", "Meta-Llama-3.1-70B-Instruct-Turbo_17_11_2024.csv")
# Define prompt for extraction
def extraction_prompt(llm_response: str):
    prompt = f"""Based on the provided response, extract the following information as a JSON object:
                - Each XBRL tag (e.g., "us-gaap:DividendsDeclaredAndPaid").
                - A list of all numerical values (e.g., [1.0, 2.0]) associated with each XBRL tag.

                Look for the "Output" section in the LLM response. Correct the formatting and structure of the JSON object.
                

                Ensure the output is in this format:
                {{
                    "XBRL_tag_1": [numerical_value_1, numerical_value_2, ...],
                    "XBRL_tag_2": [numerical_value_3, numerical_value_4, ...],
                    ...
                }}

                Extract only the relevant JSON object. Do not include any additional text.

                The response: "{llm_response}"."""
    return prompt


# Mapping function to convert labels to binary
# def map_labels(label):
#     return 1 if label == "INCLAIM" else 0

# Save progress function
def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")
    
# Normalize JSON function
def normalize_json(input_json):
    """
    Normalize extracted JSON:
    - Parse the JSON if it's a string.
    - Remove wrapping backticks or formatting artifacts.
    - Lowercase keys.
    - Ensure all numerical values in lists are floats.
    """
    try:
        # Remove triple backticks and leading/trailing whitespace
        if isinstance(input_json, str):
            input_json = input_json.strip().strip("```").strip()
            input_json = input_json.replace("'", '"')  # Replace single quotes for JSON compatibility
            data = json.loads(input_json)  # Parse cleaned string
        elif isinstance(input_json, dict):
            data = input_json
        else:
            raise ValueError("Input must be a JSON string or dictionary.")

        # Normalize keys and ensure values are lists of floats
        normalized_data = {
            str(key).strip().lower(): [
                float(val.replace(',', '')) if isinstance(val, str) and val.replace(',', '').replace('.', '').isdigit() else val
                for val in value
            ] if isinstance(value, list) else []
            for key, value in data.items()
        }
        return normalized_data
    except Exception as e:
        logger.error(f"Error normalizing JSON: {e}")
        return {}

    
# for comparing individual key-values
def compare_key_value_pairs(actual, predicted):
    """Compare key-value pairs in actual and predicted JSONs."""
    logger.info(f"Pre normalisation: {predicted}")
    logger.info(f"Pre normalisation: {actual}")
    actual = normalize_json(actual)
    predicted = normalize_json(predicted)
    # logger.info(f"Actual: {actual}")
    logger.info(f"Predicted: {predicted}")

    correct = 0
    total_actual = len(actual)
    total_predicted = len(predicted)

    # Compare each key-value pair
    for key, value in predicted.items():
        if key in actual and actual[key] == value:
            correct += 1
    accuracy = correct / (total_actual + total_predicted - correct) if (total_actual + total_predicted - correct) > 0 else 0  # Adjusted Accuracy
    precision = correct / total_predicted if total_predicted > 0 else 0
    recall = correct / total_actual if total_actual > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "correct": correct,
        "total_actual": total_actual,
        "total_predicted": total_predicted,
    }


    
    
# Evaluation function
def extract_and_evaluate_responses(args):
    results_file = INPUT_FILE_PATH

    # Load data from CSV
    df = pd.read_csv(results_file)
    llm_responses = df['llm_responses'].tolist()
    actual_labels = df['actual_labels'].tolist()
    extracted_labels = []

    # Continual save path
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / args.task
        / f"evaluation_{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize columns for storing extracted and normalized labels if not present
    if 'extracted_labels' not in df.columns:
        df['extracted_labels'] = None

    metrics = []

    for i, (llm_response, actual_label) in enumerate(zip(llm_responses, actual_labels)):
        if pd.notna(df.at[i, 'extracted_labels']):
            # Skip already processed rows
            continue

        try:
            logger.info(f"Processing response {i + 1}: {llm_response}")
            # Call the LLM to extract the JSON object
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are an expert sentence classifier focused on extracting entities."},
                    {"role": "user", "content": extraction_prompt(llm_response)},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_label = model_response.choices[0].message.content.strip() # type: ignore
            df.at[i, 'extracted_labels'] = extracted_label
            extracted_labels.append(extracted_label)

            # Compare extracted vs actual
            metric = compare_key_value_pairs(actual_label, extracted_label)
            metrics.append(metric)

            logger.info(f"Processed {i + 1}/{len(df)} responses. Metrics: {metric}")

            # Save progress after each row
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(None)

    # Aggregate metrics
    total_correct = sum(m["correct"] for m in metrics)
    total_actual = sum(m["total_actual"] for m in metrics)
    total_predicted = sum(m["total_predicted"] for m in metrics)

    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_actual if total_actual > 0 else 0
    accuracy = total_correct / (total_actual + total_predicted - total_correct) if (total_actual + total_predicted - total_correct) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # Log the evaluation metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Save evaluation metrics to DataFrame and CSV
    eval_df = pd.DataFrame({
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
    })
    eval_df.to_csv(Path(f"{str(evaluation_results_path)[:-4]}_statistics.csv"), index=False)

    # Save full results to CSV
    df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Evaluation completed. Results saved to {evaluation_results_path}")

    return df, eval_df


# Helper function for stop tokens
tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}

def tokens(model_name):
    return tokens_map.get(model_name, [])

if __name__ == "__main__":
    # Placeholder args; replace with actual argument values
    class Args:
        task = "fnxl"
        model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        date = date.today().strftime('%d_%m_%Y')
        max_tokens = 1024
        temperature = 0.0
        top_p = 0.9
        repetition_penalty = 1.0

    args = Args()
    extract_and_evaluate_responses(args)
