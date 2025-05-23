import pandas as pd
from flame.code.tokens import tokens
from litellm import completion
import re
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.logging_utils import setup_logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup logger
logger = setup_logger(
    name="convfinqa_evaluation",
    log_file=LOG_DIR / "convfinqa_evaluation.log",
    level=LOG_LEVEL,
)


# Prompt template for extracting numerical answers
def extraction_prompt(llm_response: str):
    prompt = f"""
    You will receive a response from a language model that may include a numerical answer within its text. 
    Your task is to extract and return only the main numerical value (integer, decimal, or percentage) that 
    represents the final answer. Do not include any additional text or formatting. 

    Model Response: {llm_response}

    Please respond with only one numerical value.
    """
    return prompt


# Function to extract numerical values using regex
def extract_numerical_value(text):
    match = re.search(r"(\d+(\.\d+)?%?)", text)
    return match.group(0) if match else None


# Main evaluation function
def convfinqa_evaluate(file_name, args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Note: Path definition removed - evaluate.py handles saving

    extraction_response = []
    extraction_model_response = []
    regex_extraction = []

    # Iterating over responses
    for entry in df["response"]:
        try:
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(entry)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            extraction_model_response.append(model_response)
            response_text = model_response.choices[0].message.content  # type: ignore

            extraction_response.append(response_text)

            numerical_value = extract_numerical_value(response_text)
            regex_extraction.append(numerical_value)

        except Exception as e:
            logger.error(f"Error processing response: {e}")
            extraction_model_response.append(str(e))
            extraction_response.append(None)
            regex_extraction.append(None)

    # Adding results to DataFrame
    df["extraction_model_response"] = extraction_model_response
    df["extraction_response"] = extraction_response
    df["regex_extraction"] = regex_extraction

    # Accuracy calculation
    correct_labels = df["actual_label"].tolist()
    valid_predictions = [
        (x, y) if pd.notna(x) else (x, "Error")
        for x, y in zip(correct_labels, regex_extraction)
    ]

    # Calculate metrics
    accuracy = accuracy_score(correct_labels, valid_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, valid_predictions
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
