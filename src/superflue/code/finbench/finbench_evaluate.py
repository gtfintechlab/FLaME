import pandas as pd
from litellm import completion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from superflue.utils.path_utils import get_evaluation_path
from superflue.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Define label mapping
label_mapping = {
    "LOW RISK": 0,
    "HIGH RISK": 1,
}


def extraction_prompt(llm_response: str):
    """Generate a prompt for extracting risk labels."""
    prompt = f"""Based on the following list of labels: 'HIGH RISK', 'LOW RISK', extract the most relevant label from the following response:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation."""
    return prompt


def map_label_to_number(label: str):
    """Map the extracted label to its corresponding numerical value."""
    normalized_label = label.strip().upper()  # Normalize label to uppercase
    return label_mapping.get(
        normalized_label, -1
    )  # Return -1 if the label is not found


def finbench_evaluate(file_name, args):
    """Evaluate the FinBench dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('"""')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths for results and metrics
    evaluation_results_path = get_evaluation_path(args.dataset, args.model)
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    # Initialize extracted labels
    extracted_labels = []
    correct_labels = df["actual_label"].tolist()

    # Loop through responses and extract labels
    for i, llm_response in enumerate(df["llm_responses"]):
        if pd.notna(df.at[i, "extracted_labels"]):
            continue  # Skip already processed rows

        try:
            # Generate prompt and get response
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
            mapped_label = map_label_to_number(extracted_label)

            if mapped_label == -1:
                logger.error(f"Invalid label for response {i}: {llm_response}")
            else:
                logger.info(f"Extracted label for row {i}: {mapped_label}")

            df.at[i, "extracted_labels"] = mapped_label
            extracted_labels.append(mapped_label)
            df.to_csv(evaluation_results_path, index=False)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(-1)

    # Evaluate metrics
    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, extracted_labels, average="weighted"
    )

    logger.info(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )

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
