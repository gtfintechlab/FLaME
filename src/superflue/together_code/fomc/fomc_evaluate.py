import pandas as pd
import logging
from datetime import date
from pathlib import Path
from litellm import completion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL

# Configure logging
logger = setup_logger(
    name="fomc_evaluation",
    log_file=LOG_DIR / "fomc_evaluation.log",
    level=LOG_LEVEL,
)

label_mapping = {
    "DOVISH": 0,
    "HAWKISH": 1,
    "NEUTRAL": 2,
}

def extraction_prompt(llm_response: str):
    """Generate a prompt to extract the classification label from the LLM response."""
    prompt = f'''Extract the classification label from the following LLM response. The label should be one of the following: ‘HAWKISH’, ‘DOVISH’, or ‘NEUTRAL’.
                
                Here is the LLM response to analyze:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation.'''
    return prompt

def map_label_to_number(label: str):
    """Map the extracted label to its corresponding numerical value after normalizing."""
    normalized_label = label.strip().upper()  # Normalize label to uppercase
    return label_mapping.get(normalized_label, -1)  # Return -1 if the label is not found

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def fomc_evaluate(file_name, args):
    """Evaluate FOMC dataset and return results and metrics DataFrames."""
    task = "fomc"
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file with the LLM responses
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths for saving results
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize extracted labels if not present
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    correct_labels = df["actual_labels"].tolist()
    extracted_labels = []

    for i, llm_response in enumerate(df["llm_responses"]):
        if pd.notna(df.at[i, "extracted_labels"]):
            continue

        try:
            # Generate prompt and get LLM response
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(llm_response)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_label = model_response.choices[0].message.content.strip() # type: ignore
            mapped_label = map_label_to_number(extracted_label)

            # Handle invalid labels
            if mapped_label == -1:
                logger.error(f"Invalid label for response {i}: {llm_response}")
                extracted_labels.append(-1)
            else:
                extracted_labels.append(mapped_label)

            # Update the DataFrame and save progress
            df.at[i, "extracted_labels"] = mapped_label
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(-1)

    # Calculate metrics
    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, extracted_labels, average="weighted"
    )

    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1],
    })

    # Save metrics DataFrame
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
