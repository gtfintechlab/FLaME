from datetime import date
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list
from superflue.code.extraction_prompts import refind_extraction_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
from tqdm import tqdm

# Configure logging
logger = setup_logger(
    name="refind_evaluation",
    log_file=LOG_DIR / "refind_evaluation.log",
    level=LOG_LEVEL,
)

possible_relationships = [
    'PERSON-TITLE', 'PERSON-GOV_AGY', 'PERSON-ORG', 'PERSON-UNIV',
    'ORG-ORG', 'ORG-MONEY', 'ORG-GPE', 'ORG-DATE'
]

def refind_evaluate(file_name, args):
    """Evaluate Refind dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file with the LLM responses
    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Prepare extracted labels
    extracted_labels = []
    correct_labels = df['actual_labels'].tolist()
    all_responses = df['llm_responses'].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": refind_extraction_prompt(llm_response)}]
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
                extracted_labels.append('NO-REL')
            continue

        # Process responses
        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                extracted_labels.append('NO-REL')
            extracted_label = extracted_label.replace(' ', '').replace('/', '-').replace('_', '-').upper()
            if extracted_label not in possible_relationships:
                print(f"Invalid label: {extracted_label}")
                extracted_label = 'NO-REL'
            extracted_labels.append(extracted_label)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df['extracted_labels'] = extracted_labels

    correct_labels = [label.replace(' ', '').replace('/', '-').replace('_', '-').upper() for label in correct_labels]

    # Evaluate the performance
    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(correct_labels, extracted_labels, average='weighted')

    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1]
    })

    success_rate = df["extracted_labels"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
