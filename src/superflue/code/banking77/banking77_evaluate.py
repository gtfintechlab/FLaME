import pandas as pd
from datetime import date
from litellm import completion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from superflue.config import EVALUATION_DIR
from superflue.code.prompts import banking77_list, banking77_label_map
from superflue.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Define the prompt for LLM response extraction
def extraction_prompt(llm_response: str):
    prompt = f"""Based on the following list of banking intents: {banking77_list}, extract the most relevant category from the following response:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces and underscores. Do not include any special characters or punctuation."""
    return prompt


def map_extracted_label_to_number(extracted_label: str):
    """Map the extracted label to its corresponding numerical value."""
    return banking77_label_map.get(
        extracted_label, -1
    )  # Return -1 if the label is not found


def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")


def banking77_evaluate(file_name, args):
    """Evaluate Banking 77 results and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Continual save path
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize extracted_labels column if it doesn't exist
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    correct_labels = df["actual_labels"].tolist()
    extracted_labels = []

    for i, llm_response in enumerate(df["llm_responses"]):
        if pd.notna(df.at[i, "extracted_labels"]):
            # Skip already processed rows
            continue

        try:
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(llm_response)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model),
            )
            extracted_label = model_response.choices[0].message.content.strip()  # type: ignore
            mapped_label = map_extracted_label_to_number(extracted_label)

            if mapped_label == -1:
                logger.error(f"Error processing response {i}: {llm_response}")

            df.at[i, "extracted_labels"] = mapped_label
            extracted_labels.append(mapped_label)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")

            # Save progress after each row
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(-1)

    # Evaluate performance
    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, extracted_labels, average="weighted"
    )

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1],
        }
    )

    # Save metrics DataFrame
    metrics_path = evaluation_results_path.with_name(
        f"{evaluation_results_path.stem}_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
