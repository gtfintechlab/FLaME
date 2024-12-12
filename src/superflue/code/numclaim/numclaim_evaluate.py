import pandas as pd
from pathlib import Path
from litellm import completion

# from superflue.code.tokens import tokens
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from superflue.utils.logging_utils import setup_logger
from superflue.utils.path_utils import get_evaluation_path
from superflue.config import LOG_DIR, LOG_LEVEL
import time

# Setup logger
logger = setup_logger(
    name="numclaim_evaluate",
    log_file=LOG_DIR / "numclaim_evaluate.log",
    level=LOG_LEVEL,
)


# Define prompt for extraction
def extraction_prompt(llm_response: str):
    prompt = f"""Based on the provided response, extract the following information:
                - Label the response as 'INCLAIM' if it contains a numeric value or quantitative assertion.
                - Label the response as 'OUTCLAIM' if it does not contain any numeric value or quantitative assertion.
                Provide only the label that best matches the response.
                The response: "{llm_response}"."""
    return prompt


# Mapping function to convert labels to binary
def map_labels(label):
    return 1 if label == "INCLAIM" else 0


def numclaim_evaluate(file_name, args):
    logger.info(f"Starting evaluation for Numclaim with model {args.model}...")
    # task = args.dataset.strip('"""')  # Unused variable

    # Load data from the specified file
    results_file = Path(file_name)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file {results_file} not found.")

    df = pd.read_csv(results_file)
    correct_labels = df["actual_labels"].apply(map_labels).tolist()
    llm_responses = df["llm_responses"].tolist()
    extracted_labels = []

    # Get evaluation path using new utility
    evaluation_results_path = get_evaluation_path(args.dataset, args.model)
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize the column for storing extracted labels if it doesn't exist
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    for i, llm_response in enumerate(llm_responses):
        if pd.notna(df.at[i, "extracted_labels"]):
            # Skip already processed rows
            continue

        try:
            # Correct Together API call structure
            model_response = completion(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert sentence classifier focused on identifying claims.",
                    },
                    {"role": "user", "content": extraction_prompt(llm_response)},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model),
            )
            extracted_label = model_response.choices[0].message.content.strip()  # type: ignore
            mapped_extracted_label = map_labels(extracted_label)  # Apply mapping here
            df.at[i, "extracted_labels"] = mapped_extracted_label
            extracted_labels.append(mapped_extracted_label)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")

            # Save progress after each row
            df.to_csv(evaluation_results_path, index=False)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(None)
            time.sleep(10.0)

    # Calculate evaluation metrics
    precision = precision_score(correct_labels, extracted_labels, average="binary")
    recall = recall_score(correct_labels, extracted_labels, average="binary")
    f1 = f1_score(correct_labels, extracted_labels, average="binary")
    accuracy = accuracy_score(correct_labels, extracted_labels)

    # Log the evaluation metrics
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Create metrics DataFrame with consistent format
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1],
        }
    )

    # Save metrics using consistent naming
    metrics_path = evaluation_results_path.with_name(
        f"{evaluation_results_path.stem}_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
