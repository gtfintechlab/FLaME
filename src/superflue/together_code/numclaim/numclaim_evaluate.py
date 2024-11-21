import pandas as pd
import logging
from datetime import date
from pathlib import Path
from litellm import completion 
from superflue.together_code.tokens import tokens
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from superflue.config import EVALUATION_DIR
from superflue.utils.logging_utils import setup_logger
import time

# Setup logger
logger = setup_logger(
    name="numclaim_evaluate",
    log_file=EVALUATION_DIR / "logs" / "numclaim_evaluate.log",
    level=logging.INFO,
)
# Define prompt for extraction
def extraction_prompt(llm_response: str):
    prompt = f"""Based on the provided response, extract the following information:
                - Label the response as ‘INCLAIM’ if it contains a numeric value or quantitative assertion.
                - Label the response as ‘OUTCLAIM’ if it does not contain any numeric value or quantitative assertion.
                Provide only the label that best matches the response.
                The response: "{llm_response}"."""
    return prompt

# Mapping function to convert labels to binary
def map_labels(label):
    return 1 if label == "INCLAIM" else 0

# Save progress function
def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")
def numclaim_evaluate(file_name, args):
    logger.info(f"Starting evaluation for Numclaim with model {args.model}...")
    task = args.dataset.strip('“”"')
    # Load data from the specified file
    results_file = Path(file_name)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file {results_file} not found.")

    df = pd.read_csv(results_file)
    correct_labels = df['actual_labels'].apply(map_labels).tolist()
    llm_responses = df['llm_responses'].tolist()
    extracted_labels = []

    # Continual save path
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize the column for storing extracted labels if it doesn't exist
    if 'extracted_labels' not in df.columns:
        df['extracted_labels'] = None

    for i, llm_response in enumerate(llm_responses):
        if pd.notna(df.at[i, 'extracted_labels']):
            # Skip already processed rows
            continue

        try:
            # Correct Together API call structure
            model_response = completion(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are an expert sentence classifier focused on identifying claims."},
                    {"role": "user", "content": extraction_prompt(llm_response)},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_label = model_response.choices[0].message.content.strip()  # type: ignore
            mapped_extracted_label = map_labels(extracted_label)  # Apply mapping here
            df.at[i, 'extracted_labels'] = mapped_extracted_label
            extracted_labels.append(mapped_extracted_label)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")

            # Save progress after each row
            save_progress(df, evaluation_results_path)

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

    # Save evaluation metrics to DataFrame and CSV
    eval_df = pd.DataFrame({
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "Accuracy": [accuracy]
    })
    eval_df.to_csv(Path(f"{str(evaluation_results_path)[:-4]}_statistics.csv"), index=False)

    # Save full results to CSV
    df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Evaluation completed. Results saved to {evaluation_results_path}")

    return df, eval_df
