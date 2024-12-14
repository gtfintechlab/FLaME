import pandas as pd

from litellm import completion
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from superflue.utils.logging_utils import get_logger
from superflue.utils.path_utils import get_evaluation_path

# Get logger for this module
logger = get_logger(__name__)


# Function to create the extraction prompt
def extraction_prompt(llm_response: str):
    prompt = f"""
    You are tasked with extracting the sentiment score from a response. 
    The sentiment score should be a single numeric value between -1 and 1.

    Model Response: {llm_response}

    Provide only the numerical sentiment score as the output.
    """
    return prompt


def extract_numerical_value(text):
    match = re.search(r"(-?\d+\.\d+)", text)  # Adjusted to capture decimal values
    return float(match.group(0)) if match else None


def fiqa_task1_evaluate(file_name, args):
    task = args.dataset.strip('"""')
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Get evaluation path using path utility
    evaluation_results_path = get_evaluation_path(args.dataset, args.model)
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    extraction_response = []
    extraction_model_response = []
    regex_extraction = []

    for i, entry in enumerate(df["llm_responses"]):
        try:
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(entry)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model)
            )
            extraction_model_response.append(model_response)
            response_text = model_response.choices[0].message.content  # type: ignore
            extraction_response.append(response_text)
            numerical_value = extract_numerical_value(response_text)
            regex_extraction.append(numerical_value)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extraction_response.append(None)
            regex_extraction.append(None)
            extraction_model_response.append(str(e))
            time.sleep(10.0)

    df["extraction_model_response"] = extraction_model_response
    df["extraction_response"] = extraction_response
    df["regex_extraction"] = regex_extraction

    correct_labels = df["actual_sentiment"].tolist()

    # Calculate metrics
    accuracy = accuracy_score(correct_labels, regex_extraction)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, regex_extraction
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
