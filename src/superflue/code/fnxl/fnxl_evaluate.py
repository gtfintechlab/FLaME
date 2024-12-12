import pandas as pd
import json
from litellm import completion

# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.path_utils import get_evaluation_save_path
from superflue.config import LOG_DIR, LOG_LEVEL
import time

# Configure logging
logger = setup_logger(
    name="fnxl_evaluation",
    log_file=LOG_DIR / "fnxl_evaluation.log",
    level=LOG_LEVEL,
)


def extraction_prompt(llm_response: str):
    """Generate a prompt to extract structured information from the LLM response."""
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


def normalize_json(input_json):
    """Normalize and clean up the extracted JSON."""
    try:
        if isinstance(input_json, str):
            input_json = input_json.strip().strip("```").strip()
            input_json = input_json.replace("'", '"')
            data = json.loads(input_json)
        elif isinstance(input_json, dict):
            data = input_json
        else:
            raise ValueError("Input must be a JSON string or dictionary.")

        normalized_data = {
            str(key).strip().lower(): [
                float(val.replace(",", ""))
                if isinstance(val, str)
                and val.replace(",", "").replace(".", "").isdigit()
                else val
                for val in value
            ]
            if isinstance(value, list)
            else []
            for key, value in data.items()
        }
        return normalized_data
    except Exception as e:
        logger.error(f"Error normalizing JSON: {e}")
        return {}


def compare_key_value_pairs(actual, predicted):
    """Compare key-value pairs in actual and predicted JSONs."""
    actual = normalize_json(actual)
    predicted = normalize_json(predicted)

    correct = 0
    total_actual = len(actual)
    total_predicted = len(predicted)

    for key, value in predicted.items():
        if key in actual and actual[key] == value:
            correct += 1

    accuracy = (
        correct / (total_actual + total_predicted - correct)
        if (total_actual + total_predicted - correct) > 0
        else 0
    )
    precision = correct / total_predicted if total_predicted > 0 else 0
    recall = correct / total_actual if total_actual > 0 else 0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "correct": correct,
        "total_actual": total_actual,
        "total_predicted": total_predicted,
    }


def fnxl_evaluate(file_name, args):
    """Evaluate FNXL dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('"""')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths
    evaluation_results_path = get_evaluation_save_path(args.dataset, args.model)
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    metrics = []
    for i, (llm_response, actual_label) in enumerate(
        zip(df["llm_responses"], df["actual_labels"])
    ):
        if pd.notna(df.at[i, "extracted_labels"]):
            continue

        try:
            response = completion(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(llm_response)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model),
            )
            extracted_label = response.choices[0].message.content.strip()  # type: ignore
            df.at[i, "extracted_labels"] = extracted_label

            metric = compare_key_value_pairs(actual_label, extracted_label)
            metrics.append(metric)
            logger.info(f"Processed response {i + 1}: {metric}")

            # Save progress after each row
            df.to_csv(evaluation_results_path, index=False)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            df.at[i, "extracted_labels"] = None
            metrics.append(
                {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "correct": 0,
                    "total_actual": len(normalize_json(actual_label)),
                    "total_predicted": 0,
                }
            )
            time.sleep(10.0)

    # Aggregate metrics
    total_correct = sum(m["correct"] for m in metrics)
    total_actual = sum(m["total_actual"] for m in metrics)
    total_predicted = sum(m["total_predicted"] for m in metrics)

    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_actual if total_actual > 0 else 0
    accuracy = (
        total_correct / (total_actual + total_predicted - total_correct)
        if (total_actual + total_predicted - total_correct) > 0
        else 0
    )
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    # Metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1],
        }
    )

    # Save metrics
    metrics_path = evaluation_results_path.with_name(
        f"{evaluation_results_path.stem}_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
