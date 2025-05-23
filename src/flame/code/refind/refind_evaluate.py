import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="refind_evaluation",
    log_file=LOG_DIR / "refind_evaluation.log",
    level=LOG_LEVEL,
)

possible_relationships = [
    "PERSON-TITLE",
    "PERSON-GOV_AGY",
    "PERSON-ORG",
    "PERSON-UNIV",
    "ORG-ORG",
    "ORG-MONEY",
    "ORG-GPE",
    "ORG-DATE",
]


def extraction_prompt(llm_response: str):
    """Construct the extraction prompt."""
    prompt = f"""Extract the classification label from the following LLM response. The label should be one of the following: ‘PERSON-TITLE’, ‘PERSON-GOV_AGY’, ‘PERSON-ORG’, ‘PERSON-UNIV’, ‘ORG-ORG’, ‘ORG-MONEY’, ‘ORG-GPE’, or ‘ORG-DATE’. List ‘NO-REL’ if the LLM did not output a clear answer.
                
                Here is the LLM response to analyze:
                "{llm_response}"
                Provide only the label that best matches the response, exactly as it is listed in the approved label list, with a dash (-) between words. Only output alphanumeric characters, spaces, dashes, and underscores. Do not include any special characters, quotations, or punctuation. Only output the label."""
    return prompt


def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")


def refind_evaluate(file_name, args):
    """Evaluate Refind dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    # Load the CSV file with the LLM responses
    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Prepare extracted labels
    extracted_labels = []
    correct_labels = df["actual_labels"].tolist()
    all_responses = df["llm_responses"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    # Note: Path definition removed - evaluate.py handles saving

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": extraction_prompt(llm_response)}]
            for llm_response in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            for _ in batch:
                extracted_labels.append("NO-REL")

        # Process responses
        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                extracted_labels.append("NO-REL")
            extracted_label = (
                extracted_label.replace(" ", "")
                .replace("/", "-")
                .replace("_", "-")
                .upper()
            )
            if extracted_label not in possible_relationships:
                print(f"Invalid label: {extracted_label}")
                extracted_label = "NO-REL"
            extracted_labels.append(extracted_label)

    df["extracted_labels"] = extracted_labels

    correct_labels = [
        label.replace(" ", "").replace("/", "-").replace("_", "-").upper()
        for label in correct_labels
    ]

    # Evaluate the performance
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
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1],
        }
    )

    # Note: Metrics saving removed - evaluate.py handles saving

    return df, metrics_df
