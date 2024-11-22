import pandas as pd
import logging
from datetime import date
from pathlib import Path
from superflue.together_code.tokens import tokens
from litellm import completion 
import warnings
import argparse
import re
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

def extraction_prompt(llm_response: str):
    prompt = f"""
    You will receive a response from a language model that may include a numerical answer within its text. 
    Your task is to extract and return only the main numerical value (integer, decimal, or percentage) that 
    represents the answer. Do not include any additional text or formatting. 

    Model Response: {llm_response}

    Please respond with only one numerical value.
    """
    return prompt

def extract_numerical_value(text):
    match = re.search(r"(\d+(\.\d+)?%?)", text)
    return match.group(0) if match else None

def finqa_evaluate(file_name, args):

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
    
    extraction_response = []
    extraction_model_response = []
    regex_extraction = []

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
                stop=tokens(args.model)
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

    df['extraction_model_response'] = extraction_model_response
    df['extraction_response'] = extraction_response
    df['regex_extraction']  = regex_extraction    
    
    correct_labels = df['actual_label'].tolist()

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
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1],
    })

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    df.to_csv(evaluation_results_path, index=False)

    # Save metrics DataFrame
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
