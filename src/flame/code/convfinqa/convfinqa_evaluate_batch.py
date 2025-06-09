import pandas as pd
import re
from tqdm import tqdm

from flame.utils.logging_utils import get_component_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.code.prompts.registry import get_prompt, PromptFormat
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Use component-based logger that follows the logging configuration
logger = get_component_logger("evaluation", "convfinqa")


# Function to extract numerical values using regex
def extract_numerical_value(text):
    if text is None:
        return None
    match = re.search(r"(\d+(\.\d+)?%?)", str(text))
    return match.group(0) if match else None


# Main evaluation function with batch processing
def convfinqa_evaluate(file_name, args):
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "convfinqa"
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    extraction_response = []
    extraction_model_response = []
    regex_extraction = []

    # Get all responses to process
    all_responses = df["response"].tolist()
    extraction_prompt_func = get_prompt("convfinqa", PromptFormat.EXTRACTION)

    # Prepare all prompts
    all_prompts = []
    for response in all_responses:
        if pd.notna(response):
            all_prompts.append(extraction_prompt_func(str(response)))
        else:
            all_prompts.append(None)

    logger.info(
        f"Processing {len(all_prompts)} ConvFinQA evaluations in batches of {args.batch_size}"
    )

    # Process non-null prompts in batches
    valid_indices = [i for i, prompt in enumerate(all_prompts) if prompt is not None]
    valid_prompts = [all_prompts[i] for i in valid_indices]

    if valid_prompts:
        batches = list(chunk_list(valid_prompts, args.batch_size))
        all_batch_responses = []

        for batch_idx, batch_prompts in enumerate(
            tqdm(batches, desc="Evaluating ConvFinQA responses")
        ):
            # Convert prompts to messages format for batch processing
            messages_batch = [
                [{"role": "user", "content": prompt}] for prompt in batch_prompts
            ]

            try:
                batch_responses = process_batch_with_retry(
                    args, messages_batch, batch_idx, len(batches)
                )
                all_batch_responses.extend(batch_responses)
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
                # Add None responses for failed batch
                all_batch_responses.extend([None] * len(batch_prompts))

        # Map responses back to original indices
        response_map = dict(zip(valid_indices, all_batch_responses))

        # Build final results
        for i in range(len(df)):
            if i in response_map and response_map[i]:
                model_response = response_map[i]
                extraction_model_response.append(model_response)
                response_text = model_response.choices[0].message.content  # type: ignore
                extraction_response.append(response_text)

                # Extract numerical value
                numerical_value = extract_numerical_value(response_text)
                regex_extraction.append(numerical_value)
            else:
                extraction_model_response.append(None)
                extraction_response.append(None)
                regex_extraction.append(None)
    else:
        # All responses are null
        extraction_model_response = [None] * len(df)
        extraction_response = [None] * len(df)
        regex_extraction = [None] * len(df)

    # Adding results to DataFrame
    df["extraction_model_response"] = extraction_model_response
    df["extraction_response"] = extraction_response
    df["regex_extraction"] = regex_extraction

    # Accuracy calculation
    correct_labels = df["actual_label"].tolist()

    # Filter out None predictions for metrics calculation
    valid_indices = [i for i, pred in enumerate(regex_extraction) if pred is not None]
    if valid_indices:
        valid_labels = [str(correct_labels[i]) for i in valid_indices]
        valid_predictions = [str(regex_extraction[i]) for i in valid_indices]

        # Calculate metrics
        accuracy = accuracy_score(valid_labels, valid_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_labels, valid_predictions, average="weighted", zero_division=0
        )
    else:
        logger.error("No valid predictions to calculate metrics")
        accuracy = precision = recall = f1 = 0.0

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

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}.")

    return df, metrics_df
