import pandas as pd
from datetime import date
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="finred_evaluation",
    log_file=LOG_DIR / "finred_evaluation.log",
    level=LOG_LEVEL,
)

# Define possible relationships
possible_relationships = [
    "subsidiary",
    "owned_by",
    "employer",
    "product_or_material_produced",
    "industry",
    "manufacturer",
    "developer",
    "legal_form",
    "parent_organization",
    "distribution_format",
    "chairperson",
    "location_of_formation",
    "headquarters_location",
    "operator",
    "creator",
    "currency",
    "founded_by",
    "original_broadcaster",
    "owner_of",
    "director_/_manager",
    "business_division",
    "chief_executive_officer",
    "position_held",
    "platform",
    "brand",
    "distributed_by",
    "publisher",
    "stock_exchange",
    "member_of",
]


def extraction_prompt(llm_response: str):
    """Generate a prompt to extract the classification label from the LLM response."""
    relationship_choices = ", ".join(possible_relationships)
    prompt = f"""Extract the classification label from the following LLM response. The label should be one of the following {relationship_choices}. 
    
                Pick the label out of the list that is the closest to the LLM response, but list ‘NO-REL’ if the LLM did not output a clear answer.
                
                Here is the LLM response to analyze:
                "{llm_response}"
                Provide only the label that best matches the response, exactly as it is listed in the approved label list, with an underscore (_) between words. Only output alphanumeric characters, spaces, dashes, and underscores. Do not include any special characters, quotations, asterisks, or punctuation, etc. Only output the label. Do not list an explanation or multiple labels."""
    return prompt


def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")


def finred_evaluate(file_name, args):
    """Evaluate FinRED dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    correct_labels = df["actual_labels"].tolist()
    extracted_labels = []
    all_responses = df["llm_responses"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": extraction_prompt(sentence)}]
            for sentence in sentence_batch
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in sentence_batch:
                extracted_labels.append("NO-REL")

        # Process responses
        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                extracted_label = "NO-REL"

            # Normalize and validate extracted label
            extracted_label = extracted_label.replace(" ", "")
            if extracted_label not in possible_relationships:
                logger.error(f"Invalid label: {extracted_label}")
                extracted_label = "NO-REL"

            extracted_labels.append(extracted_label)

    df["extracted_labels"] = extracted_labels

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
