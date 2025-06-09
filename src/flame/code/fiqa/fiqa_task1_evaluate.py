import pandas as pd
import re
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.code.prompts.registry import get_prompt, PromptFormat
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Setup logger
logger = setup_logger(
    name="convfinqa_evaluation",
    log_file=LOG_DIR / "convfinqa_evaluation.log",
    level=LOG_LEVEL,
)


def extract_numerical_value(text):
    match = re.search(r"(-?\d+\.\d+)", text)  # Adjusted to capture decimal values
    return float(match.group(0)) if match else None


def fiqa_task1_evaluate(file_name, args):
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "fiqa"
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Note: Path definition removed - evaluate.py handles saving

    extraction_response = []
    extraction_model_response = []
    regex_extraction = []

    all_responses = df["llm_responses"].tolist()
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        extraction_prompt_func = get_prompt("fiqa_task1", PromptFormat.EXTRACTION)
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(llm_response)}]
            for llm_response in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(batch)):
                extraction_response.append(None)
                regex_extraction.append(None)
                extraction_model_response.append(str(e))

        for response in batch_responses:
            extraction_model_response.append(response)
            try:
                response_text = response.choices[0].message.content  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_text = None
            extraction_response.append(response_text)
            numerical_value = extract_numerical_value(response_text)
            regex_extraction.append(numerical_value)

    df["extraction_model_response"] = extraction_model_response
    df["extraction_response"] = extraction_response
    df["regex_extraction"] = regex_extraction

    correct_labels = df["actual_sentiment"].tolist()

    # add else statement that just puts None if none
    correct_labels = [
        str(label) if label is not None else "Error" for label in correct_labels
    ]
    regex_extraction = [
        str(label) if label is not None else "Error" for label in regex_extraction
    ]

    # Calculate metrics
    accuracy = accuracy_score(correct_labels, regex_extraction)
    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     correct_labels, regex_extraction
    # )

    # # Log metrics
    # logger.info(f"Accuracy: {accuracy:.4f}")
    # logger.info(f"Precision: {precision:.4f}")
    # logger.info(f"Recall: {recall:.4f}")
    # logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy"],
            "Value": [accuracy],
        }
    )
    # metrics_df = pd.DataFrame({
    #     "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    #     "Value": [accuracy, precision, recall, f1],
    # })

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}.")
    # Note: File saving removed - evaluate.py handles saving

    return df, metrics_df
