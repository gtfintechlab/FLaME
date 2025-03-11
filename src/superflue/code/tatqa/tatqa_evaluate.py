import pandas as pd
import logging
from datetime import date
from pathlib import Path
from superflue.code.tokens import tokens
from litellm import completion 
import warnings
import argparse
import re
import yaml
import time

from litellm import completion 
from superflue.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
from superflue.utils.logging_utils import setup_logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


logger = setup_logger(
    name="tatqa_evaluation",
    log_file=LOG_DIR / "tatqa_evaluation.log",
    level=LOG_LEVEL,
)


def extraction_prompt(llm_response: str):
    """
    You will receive a response from a language model that may include a numerical answer within its text.
    Your task is to extract and return only the main/final answer. This could be represented as an integer,
    decimal, percentage, or text. Respond with whatever is labeled as the final answer, if that exists,
    even if that contains text. Otherwise, stick to numerical answers. Do not include any additional text or formatting.

    Model Response: {llm_response}

    Please respond with the final answer. If a final answer was not provided, respond NA.
    """
    prompt = f"""
    You will receive a response from a language model that may include a numerical answer within its text.
    Your task is to extract and return only the main/final answer. This could be represented as an integer,
    decimal, percentage, or text. Respond with whatever is labeled as the final answer, if that exists,
    even if that contains text. Otherwise, stick to numerical answers. Do not include any additional text or formatting.

    Model Response: {llm_response}

    Please respond with the final answer. If a final answer was not provided, respond NA.
    """
    return prompt


def evaluate_answer(predicted_answer: str, correct_answer: str):
    """
    You will receive two answers. Your job is to evaluate if they are exactly the same, with some caveats.
    If they are wholly different answers (e.g., 8 and 9), they are considered different.
    If the first answer is a more precise version of the second answer (e.g., units listed, more decimal points),
    they are the same.
    If the first answer can be rounded to the second answer, with the exact level of precision that the second answer uses,
    they are considered the same.
    If they cannot, they are different.
    If the answers are numbers and the first number cannot be rounded to the second number, respond with 'different'.
    For example, if the first answer is '1.02' and the second answer is '1', they are considered the same,
    but if the second answer is '1.02' and the first answer is '1.03' or '1', they are considered different.
    If the first answer is '5%' and the second answer is '5', they are considered the same.
    If the answers are the same, respond with 'correct'. If they are different, respond with 'wrong'.

    First answer: {predicted_answer}
    Second answer: {correct_answer}
    """
    prompt = f"""
    You will receive two answers. Your job is to evaluate if they are exactly the same, with some caveats.
    If they are wholly different answers (e.g., 8 and 9), they are considered different.
    If the first answer is a more precise version of the second answer (e.g., units listed, more decimal points),
    they are the same.
    If the first answer can be rounded to the second answer, with the exact level of precision that the second answer uses,
    they are considered the same.
    If they cannot, they are different.
    If the answers are numbers and the first number cannot be rounded to the second number, respond with 'different'.
    For example, if the first answer is '1.02' and the second answer is '1', they are considered the same,
    but if the second answer is '1.02' and the first answer is '1.03' or '1', they are considered different.
    If the first answer is '5%' and the second answer is '5', they are considered the same.
    If the answers are the same, respond with 'correct'. If they are different, respond with 'wrong'.

    First answer: {predicted_answer}
    Second answer: {correct_answer}
    """
    return prompt

# ------------------------------------------------------------------------------
# TATQA Evaluation Function (No Batching; Same Logic as ConvFinQA)
# ------------------------------------------------------------------------------
def tatqa_evaluate(file_name, args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    # Load the CSV data
    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

    # Prepare output path for evaluation results
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Lists to store extraction and evaluation responses
    extraction_response = []          # Final answers extracted from the model
    extraction_model_response = []    # Raw model responses for extraction
    evaluation_response = []          # Evaluation responses from the model (as text)
    evaluation_model_response = []    # Raw model responses for evaluation
    answers = []                      # Boolean flags indicating if the predicted answer is correct

    # --------------------------
    # Extraction Loop (No Batching)
    # --------------------------
    for entry in df["response"]:
        try:
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(entry)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k,
            )
        except Exception as e:
            logger.error(f"Error during extraction model completion: {e}")
            model_response = None

        try:
            response_text = model_response.choices[0].message.content  # type: ignore
        except Exception as e:
            logger.error(f"Error in extraction response: {str(e)}\nResponse: {model_response}")
            response_text = None

        extraction_model_response.append(model_response)
        extraction_response.append(response_text)

    # --------------------------
    # Evaluation Loop (Compare extracted answer vs. actual answer)
    # --------------------------
    for predicted, actual in zip(extraction_response, df["actual_answer"].tolist()):
        try:
            eval_model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": evaluate_answer(predicted, actual)}],
                max_tokens=args.max_tokens * 2,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k,
            )
        except Exception as e:
            logger.error(f"Error during evaluation model completion: {e}")
            eval_model_response = None

        try:
            eval_response_text = eval_model_response.choices[0].message.content.lower()  # type: ignore
        except Exception as e:
            logger.error(f"Error in evaluation response: {str(e)}\nResponse: {eval_model_response}")
            eval_response_text = None

        evaluation_model_response.append(eval_model_response)
        evaluation_response.append(eval_response_text)

        # Determine correctness based on "correct" vs. "wrong" logic
        if eval_response_text is not None:
            find_correct = eval_response_text.find("correct")
            find_wrong = eval_response_text.find("wrong")
            answers.append(find_correct != -1 and (find_wrong == -1 or find_correct < find_wrong))
        else:
            answers.append(False)

    # Add new columns to the DataFrame
    df['extraction_model_response'] = extraction_model_response
    df['extraction_response'] = extraction_response
    df['evaluation_model_response'] = evaluation_model_response
    df['evaluation_response'] = evaluation_response
    df['final_answer'] = answers

    # Compute Accuracy
    accuracy = len([a for a in answers if a]) / len(answers) if answers else 0.0
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({"metric": ["accuracy"], "value": [accuracy]})

    # Save results and metrics to CSV files
    df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Evaluation results saved to {evaluation_results_path}")

    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df

# ------------------------------------------------------------------------------
# Main Function (Loads config from YAML and calls evaluation)
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TATQA Evaluation")
    parser.add_argument("--config", type=str, default="default.yaml", help="Path to the YAML config file")
    args_cli = parser.parse_args()

    # Load configuration from YAML
    with open(args_cli.config, "r") as f:
        config = yaml.safe_load(f)

    # Convert the config dictionary into a Namespace for attribute access
    args_namespace = argparse.Namespace(**config)

    # Run evaluation
    df_results, metrics_df = tatqa_evaluate(args_namespace.file_name, args_namespace)
    print(f"Final Accuracy: {metrics_df['value'][0]:.4f}")

if __name__ == "__main__":
    main()
