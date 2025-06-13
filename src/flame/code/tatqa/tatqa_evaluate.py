import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.logging_utils import get_component_logger

# Use component-based logger that follows the logging configuration
logger = get_component_logger("evaluation", "tatqa")


# Function to generate the evaluation prompt
def evaluation_prompt(llm_response: str, actual_answer: str):
    prompt = f"""
    The correct answer is {actual_answer}. Based on the model's response, extract the numerical value closest to the correct label.
    Return only the number and no additional words, punctuation, or text. For example, 13 or 90%.
    If there are multiple numbers, return only the most relevant one.

    Response: {llm_response}
    """
    return prompt


def tatqa_evaluate(file_name, args):
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "tatqa"
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Initialize list for storing evaluation results
    evaluation_results = []
    extraction_model_response = []

    # Prepare all prompts
    all_prompts = []
    for llm_response, actual_answer in zip(df["response"], df["actual_answer"]):
        if pd.notna(llm_response):
            all_prompts.append(evaluation_prompt(str(llm_response), str(actual_answer)))
        else:
            all_prompts.append(None)

    logger.info(
        f"Processing {len(all_prompts)} TATQA evaluations in batches of {args.batch_size}"
    )

    # Process non-null prompts in batches
    valid_indices = [i for i, prompt in enumerate(all_prompts) if prompt is not None]
    valid_prompts = [all_prompts[i] for i in valid_indices]

    if valid_prompts:
        batches = list(chunk_list(valid_prompts, args.batch_size))
        all_batch_responses = []

        for batch_idx, batch_prompts in enumerate(
            tqdm(batches, desc="Evaluating TATQA responses")
        ):
            # Convert prompts to messages format for batch processing
            messages_batch = [
                [{"role": "user", "content": prompt}] for prompt in batch_prompts
            ]
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, len(batches)
            )
            all_batch_responses.extend(batch_responses)

        # Map responses back to original indices
        response_map = dict(zip(valid_indices, all_batch_responses))

        # Build final results
        for i in range(len(df)):
            if i in response_map and response_map[i]:
                extraction_model_response.append(response_map[i])
                response_text = response_map[i].choices[0].message.content  # type: ignore
                evaluation_results.append(response_text)
            else:
                extraction_model_response.append(None)
                evaluation_results.append(None)
    else:
        # All responses are null
        extraction_model_response = [None] * len(df)
        evaluation_results = [None] * len(df)

    # Update DataFrame with extracted results
    df["extracted_labels"] = evaluation_results

    correct_labels = df["actual_answer"].tolist()

    # Filter out None values for metrics calculation
    valid_indices = [
        i for i, result in enumerate(evaluation_results) if result is not None
    ]
    valid_labels = [correct_labels[i] for i in valid_indices]
    valid_results = [evaluation_results[i] for i in valid_indices]

    if not valid_results:
        logger.error("No valid evaluation results to calculate metrics")
        accuracy = precision = recall = f1 = 0.0
    else:
        # Convert to strings for comparison
        valid_labels = [str(label) for label in valid_labels]
        valid_results = [str(result).strip() for result in valid_results]

        valid_labels_array = np.array(valid_labels)
        valid_results_array = np.array(valid_results)
        accuracy = accuracy_score(valid_labels_array, valid_results_array)
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_labels_array, valid_results_array, average="weighted", zero_division=0
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

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}.")

    return df, metrics_df
