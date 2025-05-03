import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm
from superflue.code.extraction_prompts import causal_detection_extraction_prompt
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
import ast

logger = setup_logger(
    name="causal_detection_evaluate",
    log_file=LOG_DIR / "causal_detection_evaluate.log",
    level=LOG_LEVEL,
)


def adjust_tags(row):
    actual = row["actual_tags"]
    predicted = row["extracted_tags"]
    if len(predicted) > len(actual):
        return predicted[: len(actual)]
    elif len(predicted) < len(actual):
        return predicted + ["NA"] * (len(actual) - len(predicted))
    else:
        return predicted


def causal_detection_evaluate(file_name, args):
    """Evaluate causal detection results and return results and metrics DataFrames."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    extracted_tags = []

    all_responses = df["llm_responses"].tolist()

    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": causal_detection_extraction_prompt(response)}]
            for response in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add None values for failed batch
            for _ in batch:
                extracted_tags.append([])

        # Process responses
        for response in batch_responses:
            try:
                extracted_list = response.choices[0].message.content.strip()  # type: ignore
                extracted_list = extracted_list.replace("‘", "'").replace("’", "'")
                extracted_list = extracted_list[
                    extracted_list.find("[") : max(
                        extracted_list.rfind("]"), len(extracted_list) - 1
                    )
                    + 1
                ]
                try:
                    eval(extracted_list)
                    if extracted_list.count("[") > 1:
                        extracted_list = "[]"
                except Exception:
                    extracted_list = "[]"
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                extracted_list = "[]"
            extracted_tags.append(extracted_list)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df["extracted_tags"] = extracted_tags
    # Evaluate performance

    df["extracted_tags"] = df["extracted_tags"].apply(ast.literal_eval)
    df["actual_tags"] = df["actual_tags"].apply(ast.literal_eval)

    df["adjusted_extracted_tags"] = df.apply(adjust_tags, axis=1)

    df["length_match"] = df["adjusted_extracted_tags"].notnull()

    df["row_accuracy"] = df.apply(
        lambda row: accuracy_score(row["actual_tags"], row["adjusted_extracted_tags"])
        if row["length_match"]
        else 0.0,
        axis=1,
    )

    valid_rows = df[df["length_match"]]

    flat_actual = [tag for tags in valid_rows["actual_tags"] for tag in tags]
    flat_predicted = [
        tag for tags in valid_rows["adjusted_extracted_tags"] for tag in tags
    ]

    labels = ["B-CAUSE", "I-CAUSE", "B-EFFECT", "I-EFFECT", "O"]
    print("Token Classification Report:")
    print(classification_report(flat_actual, flat_predicted, labels=labels))

    accuracy = accuracy_score(flat_actual, flat_predicted)
    print(f"Overall Token-Level Accuracy: {accuracy:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_actual, flat_predicted, average="weighted"
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

    success_rate = df["extracted_tags"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
