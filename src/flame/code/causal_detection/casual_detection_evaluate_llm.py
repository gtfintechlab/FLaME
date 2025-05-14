import pandas as pd
from datetime import date
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
from tqdm import tqdm
from litellm.types.utils import (
    ModelResponse,
    Choices,
    Message,
    Usage,
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
)
import ast

# Configure logging
logger = setup_logger(
    name="causal_detection_evaluate",
    log_file=LOG_DIR / "causal_detection_evaluate.log",
    level=LOG_LEVEL,
)


# Define the prompt for LLM response extraction
def extraction_prompt(llm_response: str):
    prompt = f"""Given the following output from a language model, extract the entire list of tokens. The allowed tokens are 'O', 'I-CAUSE', 'B-CAUSE', 'I-EFFECT', and 'B-EFFECT'.
                The list should only contain these tokens and should be enclosed in brackets. Each token should be a string and surrounded by quotations ('').
                Extract all tokens that were found and output them in the exact order they were originally written. Only output tokens from the input, do not add any tokens. If no tokens were found, output an empty list.
                Only output a list of tokens enclosed in brackets, do not include any additional text or formatting.
                Response: {llm_response}"""
    return prompt


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

    # Continual save path
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize extracted_labels column if it doesn't exist
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    extracted_tags = []

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

    df["llm_responses"] = df["llm_responses"].apply(
        lambda x: x[(x.find("</think>") + 8) :]
    )

    all_responses = df["llm_responses"].tolist()

    # Create batches for processing
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": extraction_prompt(response)}]
            for response in batch
        ]

        try:
            # Process batch with retry logic
            # Using the process_batch_with_retry function from batch_utils
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
                extracted_list.replace("‘", "'").replace("’", "'")
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
            logger.debug(f"Processed {len(extracted_tags)}/{len(df)} responses.")

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df["extracted_tags"] = extracted_tags
    # Evaluate performance

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
    print("Token Classification Report:")
    print(classification_report(flat_actual, flat_predicted, labels=labels))

    accuracy = accuracy_score(flat_actual, flat_predicted)
    print(f"Overall Token-Level Accuracy: {accuracy:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_actual, flat_predicted, average="weighted"
    )

    logger.info(
        f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}"
    )
    df.to_csv(evaluation_results_path, index=False)

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

    # Save metrics DataFrame
    metrics_path = evaluation_results_path.with_name(
        f"{evaluation_results_path.stem}_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
