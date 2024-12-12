import pandas as pd
import time
import re
from litellm import completion
from superflue.utils.logging_utils import setup_logger
from superflue.utils.path_utils import get_evaluation_save_path
from superflue.config import LOG_DIR, LOG_LEVEL
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Setup logger
logger = setup_logger(
    name="finqa_evaluation",
    log_file=LOG_DIR / "finqa_evaluation.log",
    level=LOG_LEVEL,
)


def extraction_prompt(llm_response: str) -> str:
    """Generate a prompt to extract numerical values from model responses."""
    prompt = f"""
    You will receive a response from a language model that may include a numerical answer within its text. 
    Your task is to extract and return only the main numerical value (integer, decimal, or percentage) that 
    represents the answer. Do not include any additional text or formatting. 

    Model Response: {llm_response}

    Please respond with only one numerical value.
    """
    return prompt


def extract_numerical_value(text: str) -> str | None:
    """Extract numerical value from text using regex pattern matching."""
    match = re.search(r"(\d+(\.\d+)?%?)", text)
    return match.group(0) if match else None


def finqa_evaluate(file_name: str, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate FinQA dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('"""')
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    # Load data
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths using consistent utility
    evaluation_results_path = get_evaluation_save_path(args.dataset, args.model)
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    extraction_response = []
    extraction_model_response = []
    regex_extraction = []

    for entry in tqdm(df["llm_responses"], desc="Processing responses"):
        try:
            # Generate prompt and get response
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

            # Log and process the response
            logger.debug(f"Model response: {model_response}")
            extraction_model_response.append(model_response)
            response_text = model_response.choices[0].message.content  # type: ignore
            extraction_response.append(response_text)

            numerical_value = extract_numerical_value(response_text)
            regex_extraction.append(numerical_value)

        except Exception as e:
            logger.error(f"Error processing response: {e}")
            extraction_response.append(None)
            regex_extraction.append(None)
            extraction_model_response.append(str(e))
            time.sleep(10.0)

    # Update DataFrame with results
    df["extraction_model_response"] = extraction_model_response
    df["extraction_response"] = extraction_response
    df["regex_extraction"] = regex_extraction

    correct_labels = df["actual_label"].tolist()

    # Calculate metrics
    accuracy = accuracy_score(correct_labels, regex_extraction)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, regex_extraction, average="weighted"
    )

    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame with consistent format
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1],
        }
    )

    # Save results
    df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Results saved to {evaluation_results_path}")

    # Save metrics using consistent naming
    metrics_path = evaluation_results_path.with_name(
        f"{evaluation_results_path.stem}_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
