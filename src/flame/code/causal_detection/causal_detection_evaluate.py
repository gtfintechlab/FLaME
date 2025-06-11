import ast

import pandas as pd
from litellm.types.utils import (
    Choices,
    CompletionTokensDetailsWrapper,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from flame.code.prompts.registry import PromptFormat, get_prompt
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.logging_utils import setup_logger

# Configure logging
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
    # support legacy args.dataset for tests, prefer args.task
    task = (
        getattr(args, "task", None)
        or getattr(args, "dataset", None)
        or "causal_detection"
    )
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load the CSV file
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Initialize extracted_labels column if it doesn't exist
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    # PHASE 1: Prepare data and extract responses
    logger.info("Phase 1: Preparing data and extracting LLM responses...")

    type_dict = {
        "ModelResponse": ModelResponse,
        "Choices": Choices,
        "Message": Message,
        "Usage": Usage,
        "CompletionTokensDetailsWrapper": CompletionTokensDetailsWrapper,
        "PromptTokensDetailsWrapper": PromptTokensDetailsWrapper,
    }
    df["complete_responses"] = df["complete_responses"].apply(
        lambda x: eval(x, type_dict)
    )
    df["llm_responses"] = df["complete_responses"].apply(
        lambda x: x.choices[0].message.content
    )

    # Remove think tags if present
    df["llm_responses"] = df["llm_responses"].apply(
        lambda x: x[(x.find("</think>") + 8) :] if "</think>" in x else x
    )

    all_responses = df["llm_responses"].tolist()

    # PHASE 2: Make all API calls to extract tags
    logger.info(f"Phase 2: Making {len(all_responses)} API calls to extract tags...")

    extracted_tags = []  # Initialize the list to store results

    # Create batches for processing
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Making API calls")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch
        extraction_prompt_func = get_prompt("causal_detection", PromptFormat.EXTRACTION)
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(response)}]
            for response in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            # Store raw responses
            for response in batch_responses:
                try:
                    extracted_content = response.choices[0].message.content.strip()
                    extracted_tags.append(extracted_content)
                except Exception as e:
                    logger.debug(f"Error extracting content from response: {str(e)}")
                    extracted_tags.append("[]")

        except Exception as e:
            logger.debug(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add empty results for failed batch
            for _ in batch:
                extracted_tags.append("[]")

        pbar.set_description(f"Completed batch {batch_idx + 1}/{total_batches}")

    logger.info(f"Phase 2 complete. Extracted {len(extracted_tags)} tag sets.")

    # PHASE 3: Post-process all extracted tags
    logger.info("Phase 3: Post-processing extracted tags...")

    # Process extracted tags
    processed_tags = []
    for extracted_list in extracted_tags:
        try:
            # Clean up the extracted string
            extracted_list = extracted_list.replace("'", "'").replace("'", "'")

            # Find the list boundaries
            start_idx = extracted_list.find("[")
            end_idx = extracted_list.rfind("]")

            if start_idx != -1 and end_idx != -1:
                extracted_list = extracted_list[start_idx : end_idx + 1]
            else:
                extracted_list = "[]"

            # Validate the list
            try:
                eval(extracted_list)
                if extracted_list.count("[") > 1:
                    extracted_list = "[]"
            except Exception:
                extracted_list = "[]"

            processed_tags.append(extracted_list)

        except Exception as e:
            logger.debug(f"Error processing extracted tag: {str(e)}")
            processed_tags.append("[]")

    df["extracted_tags"] = processed_tags

    # PHASE 4: Evaluate performance
    logger.info("Phase 4: Evaluating performance...")

    df["extracted_tags"] = df["extracted_tags"].apply(ast.literal_eval)
    df["actual_tags"] = df["actual_tags"].apply(ast.literal_eval)

    df["adjusted_extracted_tags"] = df.apply(adjust_tags, axis=1)

    df["length_match"] = df["adjusted_extracted_tags"].notnull()

    df["row_accuracy"] = df.apply(
        lambda row: (
            accuracy_score(row["actual_tags"], row["adjusted_extracted_tags"])
            if row["length_match"]
            else 0.0
        ),  # type: ignore
        axis=1,
    )  # type: ignore

    valid_rows = df[df["length_match"]]

    flat_actual = [tag for tags in valid_rows["actual_tags"] for tag in tags]
    flat_predicted = [
        tag for tags in valid_rows["adjusted_extracted_tags"] for tag in tags
    ]

    labels = ["B-CAUSE", "I-CAUSE", "B-EFFECT", "I-EFFECT", "O"]
    logger.info("Token Classification Report:")
    logger.info(classification_report(flat_actual, flat_predicted, labels=labels))

    accuracy = accuracy_score(flat_actual, flat_predicted)
    logger.info(f"Overall Token-Level Accuracy: {accuracy:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_actual, flat_predicted, average="weighted"
    )

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}.")

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

    return df, metrics_df
