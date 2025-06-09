import pandas as pd
from flame.code.tokens import tokens
from litellm import completion
from tqdm import tqdm
from flame.utils.logging_utils import get_component_logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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


# Function for extracting and evaluating responses


def tatqa_evaluate(file_name, args):
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "tatqa"
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Note: Path definition removed - evaluate.py handles saving

    # Initialize list for storing evaluation results
    evaluation_results = []
    extraction_model_response = []

    # Iterate over each response and evaluate
    for i, (llm_response, actual_answer) in enumerate(
        tqdm(
            zip(df["response"], df["actual_answer"]),
            total=len(df),
            desc="Evaluating TATQA responses",
        )
    ):
        try:
            model_response = completion(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": evaluation_prompt(llm_response, actual_answer),
                    }
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extraction_model_response.append(model_response)
            response_text = model_response.choices[0].message.content  # type: ignore
            evaluation_results.append(response_text)
            logger.debug(f"Processed {i + 1}/{len(df)} responses.")
        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extraction_model_response.append(str(e))
            evaluation_results.append(None)
            # time.sleep(10.0)  # Removed sleep for better performance
            pass

        # Update DataFrame with extracted results after each iteration
        df["extracted_labels"] = evaluation_results

        # Note: Progress saving removed - evaluate.py handles saving
        logger.debug(f"Processed iteration {i + 1}/{len(df)}")

    correct_labels = df["actual_answer"].tolist()

    # Calculate metrics
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
        accuracy = accuracy_score(valid_labels, valid_results)
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_labels, valid_results, average="weighted", zero_division=0
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
    # Note: File saving removed - evaluate.py handles saving

    return df, metrics_df
