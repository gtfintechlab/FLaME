import pandas as pd
from datetime import date
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.code.extraction_prompts import finred_extraction_prompt, finred_possible_relationships
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL

# Configure logging
logger = setup_logger(
    name="finred_evaluation",
    log_file=LOG_DIR / "finred_evaluation.log",
    level=LOG_LEVEL,
)

def finred_evaluate(file_name, args):
    """Evaluate FinRED dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    correct_labels = df["actual_labels"].tolist()
    extracted_labels = []
    all_responses = df["llm_responses"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)
    
    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": finred_extraction_prompt(sentence)}]
            for sentence in sentence_batch
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in sentence_batch:
                extracted_labels.append('NO-REL')
            continue

        # Process responses
        for response in batch_responses:
            try:
                extracted_label = response.choices[0].message.content.strip() # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                extracted_label = 'NO-REL'
                
            # Normalize and validate extracted label
            extracted_label = extracted_label.replace(' ', '')
            if extracted_label not in finred_possible_relationships:
                logger.error(f"Invalid label: {extracted_label}")
                extracted_label = 'NO-REL'

            extracted_labels.append(extracted_label)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df['extracted_labels'] = extracted_labels
    
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

    success_rate = df["extracted_labels"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
