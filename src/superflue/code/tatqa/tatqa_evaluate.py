import pandas as pd
from datetime import date

# from superflue.code.tokens import tokens
from litellm import completion
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
from superflue.utils.logging_utils import setup_logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

# Setup logger
logger = setup_logger(
    name="convfinqa_evaluation",
    log_file=LOG_DIR / "convfinqa_evaluation.log",
    level=LOG_LEVEL,
)


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
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Output path for evaluation results
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize list for storing evaluation results
    evaluation_results = []
    extraction_model_response = []

    # Iterate over each response and evaluate
    for i, (llm_response, actual_answer) in enumerate(
        zip(df["response"], df["actual_answer"])
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
                # stop=tokens(args.model)
            )
            extraction_model_response.append(model_response)
            response_text = model_response.choices[0].message.content  # type: ignore
            evaluation_results.append(response_text)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extraction_model_response.append(str(e))
            evaluation_results.append(None)
            time.sleep(10.0)

        # Update DataFrame with extracted results after each iteration
        df["extracted_labels"] = evaluation_results

        # Save the updated DataFrame to CSV after each iteration
        df.to_csv(evaluation_results_path, index=False)
        logger.info(f"CSV updated at iteration {i + 1}/{len(df)}")

    correct_labels = df["actual_answer"].tolist()

    # Calculate metrics
    accuracy = accuracy_score(correct_labels, evaluation_results)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, evaluation_results
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

    logger.info(
        f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}"
    )
    df.to_csv(evaluation_results_path, index=False)

    # Save metrics DataFrame
    metrics_path = evaluation_results_path.with_name(
        f"{evaluation_results_path.stem}_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
