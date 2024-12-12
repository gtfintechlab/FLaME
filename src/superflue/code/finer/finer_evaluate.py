import pandas as pd
import json
from litellm import completion
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.path_utils import get_evaluation_path
from superflue.config import LOG_DIR, LOG_LEVEL

# Configure logging
logger = setup_logger(
    name="finer_evaluation",
    log_file=LOG_DIR / "finer_evaluation.log",
    level=LOG_LEVEL,
)


def extraction_prompt_finer(llm_response: str):
    """Generate a prompt to extract numeric labels for named entity recognition."""
    prompt = f"""For each token in the following response, map the named entity labels to these numeric values:
                    - "O" (Other): 0
                    - "PER_B" (Person_B): 1
                    - "PER_I" (Person_I): 2
                    - "LOC_B" (Location_B): 3
                    - "LOC_I" (Location_I): 4
                    - "ORG_B" (Organisation_B): 5
                    - "ORG_I" (Organisation_I): 6

                Provide only the list of integer labels, in the format:
                [0, 1, 0, ...]

                Do not include any additional text, explanations, or formatting other than a plain list.

                LLM response:
                "{llm_response}"."""
    return prompt


def clean_extracted_list(response: str) -> str:
    """Clean and format the extracted list response."""
    # Remove any text before and after the list
    start_idx = response.find("[")
    end_idx = response.rfind("]") + 1
    if start_idx == -1 or end_idx == 0:
        return "[]"
    return response[start_idx:end_idx]


def finer_evaluate(file_name, args):
    """Evaluate FINER dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('"""')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths
    evaluation_results_path = get_evaluation_path(args.dataset, args.model)
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize columns
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    correct_labels = (
        df["actual_labels"]
        .apply(lambda x: json.loads(x) if pd.notna(x) else [])
        .tolist()
    )
    extracted_labels = []

    for i, llm_response in enumerate(df["llm_responses"]):
        if pd.notna(df.at[i, "extracted_labels"]):
            continue

        try:
            # Generate prompt and get response
            response = completion(
                model=args.model,
                messages=[
                    {"role": "user", "content": extraction_prompt_finer(llm_response)}
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model),
            )
            extracted_list = response.choices[0].message.content.strip()  # type: ignore
            cleaned_response = clean_extracted_list(extracted_list)
            extracted_tokens = json.loads(cleaned_response)

            # Update DataFrame
            df.at[i, "extracted_labels"] = extracted_tokens
            extracted_labels.append(extracted_tokens)
            df.to_csv(evaluation_results_path, index=False)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append([])

    # Flatten the lists for metrics
    flat_correct_labels = [label for sublist in correct_labels for label in sublist]
    flat_extracted_labels = [label for sublist in extracted_labels for label in sublist]

    # Calculate metrics
    precision = precision_score(
        flat_correct_labels, flat_extracted_labels, average="macro", zero_division=0
    )
    recall = recall_score(
        flat_correct_labels, flat_extracted_labels, average="macro", zero_division=0
    )
    f1 = f1_score(
        flat_correct_labels, flat_extracted_labels, average="macro", zero_division=0
    )
    accuracy = accuracy_score(flat_correct_labels, flat_extracted_labels)

    # Log metrics
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
