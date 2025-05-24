import pandas as pd
import re
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.code.prompts.registry import get_prompt, PromptFormat
from tqdm import tqdm

logger = setup_logger(
    name="convfinqa_evaluation",
    log_file=LOG_DIR / "convfinqa_evaluation.log",
    level=LOG_LEVEL,
)


def evaluate_answer(predicted_answer: str, correct_answer: str):
    prompt = f"""
    You will receive two answers. Your job is to evaluate if they are exactly the same, with some caveats. 
    If they are wholly different answers (eg: 8 and 9), they are considered different.
    If the first answer is a more precise version of the second answer (eg: units listed, more decimal points reported, etc), they are the same.
    If the first answer can be rounded to the second answer, with the exact level of precision that the second answer uses, they are considered the same. If they cannot, they are different.
    If the answers are numbers and the first number cannot be rounded to the second number, respond with 'different'.
    For example, if the first answer is '1.02' and the second answer is '1', they are considered the same,
    but if the second answer is '1.02' and the first answer is '1.03' or '1', they are considered different.
    If the first answer is '5%' and the second answer is '5', they are considered the same.
    If the answers are the same, respond with 'correct'. If they are different, respond with 'wrong'.
    First answer: {predicted_answer}. Second answer: {correct_answer}
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

    # Note: Path definition removed - evaluate.py handles saving

    extraction_response = []
    extraction_model_response = []
    evaluation_response = []
    evaluation_model_response = []

    answers = []

    all_responses = df["response"].tolist()
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        extraction_prompt_func = get_prompt("qa", PromptFormat.EXTRACTION)
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(response)}]
            for response in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches, args.max_tokens
            )

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for _ in range(len(batch)):
                extraction_response.append(None)
                extraction_model_response.append(str(e))

        for response in batch_responses:
            extraction_model_response.append(response)
            try:
                response_text = response.choices[0].message.content  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_text = None
            extraction_response.append(response_text)

    all_responses = [
        (response, actual_label)
        for response, actual_label in zip(
            extraction_response, df["actual_label"].tolist()
        )
    ]
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": evaluate_answer(predicted, actual)}]
            for predicted, actual in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches, args.max_tokens * 2
            )

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for _ in range(len(batch)):
                evaluation_response.append(None)
                evaluation_model_response.append(str(e))

        for response in batch_responses:
            evaluation_model_response.append(response)
            try:
                response_text = response.choices[0].message.content.lower()  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_text = None
            evaluation_response.append(response_text)
            find_correct = response_text.find("correct")  # type: ignore
            find_wrong = response_text.find("wrong")  # type: ignore
            answers.append(
                find_correct != -1 and (find_wrong == -1 or find_correct < find_wrong)
            )

    df["extraction_model_response"] = extraction_model_response
    df["extraction_response"] = extraction_response
    df["evaluation_model_response"] = evaluation_model_response
    df["evaluation_response"] = evaluation_response
    df["final_answer"] = answers

    # Calculate metrics
    accuracy = len([answer for answer in answers if answer]) / len(answers)

    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "metric": ["accuracy"],
            "value": [accuracy],
        }
    )

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}.")
    # Note: File saving removed - evaluate.py handles saving

    return df, metrics_df
