import json
from datetime import date
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from litellm import completion
from pathlib import Path
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
import time

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
    "Past_News": {"0": 0, "1": 1}
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

def extraction_prompt(llm_response: str):
    """Generate a prompt to extract the relevant information from the LLM response."""
    prompt = f"""Extract the relevant information from the following LLM response and provide a score of 0 or 1 for each attribute based on the content. Format your output as a JSON object with these keys:
    - "Price_or_Not"
    - "Direction_Up"
    - "Direction_Down"
    - "Direction_Constant"
    - "Past_Price"
    - "Future_Price"
    - "Past_News"
    Only output the keys and values in the JSON object. Do not include any additional text.
    LLM Response:
    "{llm_response}" """
    return prompt

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

    # Paths
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize extracted labels if not present
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    correct_labels = df[[
        "price_or_not", "direction_up", "direction_down",
        "direction_constant", "past_price", "future_price", "past_news"
    ]].values.tolist()
    extracted_labels = []

    for i, llm_response in enumerate(df["llm_responses"]):
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
                stop=tokens(args.model)
            )
            raw_response = response.choices[0].message.content.strip()  # type: ignore # Extract raw response
            preprocessed_response = preprocess_llm_response(raw_response)
            if not preprocessed_response:
                raise ValueError(f"Preprocessing failed for response: {raw_response}")

            # Validate and parse JSON
            try:
                extracted_label_json = json.loads(preprocessed_response)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in response {i}: {e}. response: {preprocessed_response}")
                extracted_labels.append([-1] * 7)  # Assign default invalid labels
                df.at[i, "extracted_labels"] = [-1] * 7
                save_progress(df, evaluation_results_path)
                continue

            # Map the extracted labels to numeric values
            mapped_labels = [
                map_label_to_number(str(extracted_label_json.get("Price_or_Not", "")), "Price_or_Not"),
                map_label_to_number(str(extracted_label_json.get("Direction_Up", "")), "Direction_Up"),
                map_label_to_number(str(extracted_label_json.get("Direction_Down", "")), "Direction_Down"),
                map_label_to_number(str(extracted_label_json.get("Direction_Constant", "")), "Direction_Constant"),
                map_label_to_number(str(extracted_label_json.get("Past_Price", "")), "Past_Price"),
                map_label_to_number(str(extracted_label_json.get("Future_Price", "")), "Future_Price"),
                map_label_to_number(str(extracted_label_json.get("Past_News", "")), "Past_News"),
            ]

            # Update the DataFrame and save progress
            df.at[i, "extracted_labels"] = mapped_labels
            extracted_labels.append(mapped_labels)
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error at row {i}: {e}")
            extracted_labels.append([-1] * 7)
            time.sleep(10.0)

    # Metrics
    correct_predictions = [list(map(int, labels)) for labels in correct_labels]
    accuracy = accuracy_score(correct_predictions, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_predictions, extracted_labels, average="weighted"
    )

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1]
    })

    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    logger.info(f"Metrics saved to {metrics_path}")
    return df, metrics_df
