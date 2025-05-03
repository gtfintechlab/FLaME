import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.code.extraction_prompts import causal_classifciation_extraction_prompt
from flame.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm

logger = setup_logger(
    name="causal_classification_evaluation",
    log_file=LOG_DIR / "causal_classification_evaluation.log",
    level=LOG_LEVEL,
)


def normalize_response(response):
    """Normalize the LLM response to extract the predicted label."""
    try:
        response = response.strip()
        if response.isdigit():
            return int(response)
        elif "0" in response:
            return 0
        elif "1" in response:
            return 1
        elif "2" in response:
            return 2
        else:
            raise ValueError(f"Invalid response format: {response}")
    except Exception as e:
        logger.error(f"Error normalizing response: {e}")
        return None


def causal_classification_evaluate(file_name, args):
    """Evaluate causal classification results with label extraction and comparison."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    all_responses = df["llm_responses"].tolist()
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    extracted_labels = []
    logger.info(f"Processing {len(df)} rows in {total_batches} batches.")

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [
                {
                    "role": "user",
                    "content": causal_classifciation_extraction_prompt(response),
                }
            ]
            for response in batch
        ]
        logger.info(f"Generated messages for batch {messages_batch}.")

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            for _ in batch:
                extracted_labels.append(None)
            continue

        for response in batch_responses:
            try:
                llm_response = response.choices[0].message.content.strip()  # type: ignore
                predicted_label = normalize_response(llm_response)

                if predicted_label is not None:
                    extracted_labels.append(predicted_label)
                else:
                    extracted_labels.append(None)

            except Exception as e:
                logger.error(f"Error processing response for batch {batch_idx}: {e}")
                extracted_labels.append(None)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df["extracted_labels"] = extracted_labels

    valid_indices = [
        i for i in range(len(extracted_labels)) if extracted_labels[i] is not None
    ]
    filtered_predicted = [extracted_labels[i] for i in valid_indices]
    filtered_actual = [df.at[i, "actual_labels"] for i in valid_indices]

    precision = precision_score(filtered_actual, filtered_predicted, average="macro")
    recall = recall_score(filtered_actual, filtered_predicted, average="macro")
    f1 = f1_score(filtered_actual, filtered_predicted, average="macro")
    accuracy = accuracy_score(filtered_actual, filtered_predicted)

    metrics_df = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
            "Value": [precision, recall, f1, accuracy],
        }
    )

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    success_rate = df["extracted_labels"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
