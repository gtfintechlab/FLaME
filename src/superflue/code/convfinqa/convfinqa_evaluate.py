import pandas as pd
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list
from superflue.config import LOG_DIR, LOG_LEVEL
from superflue.utils.logging_utils import setup_logger
from superflue.code.extraction_prompts import qa_extraction_prompt, qa_evaluate_answer
from tqdm import tqdm

logger = setup_logger(
    name="convfinqa_evaluation",
    log_file=LOG_DIR / "convfinqa_evaluation.log",
    level=LOG_LEVEL,
)


def convfinqa_evaluate(file_name, args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded data from {file_name} for evaluation.")

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
        messages_batch = [
            [{"role": "user", "content": qa_extraction_prompt(response)}]
            for response in batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for _ in range(len(batch)):
                extraction_response.append(None)
                extraction_model_response.append(str(e))
            continue

        for response in batch_responses:
            try:
                response_text = response.choices[0].message.content  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_text = None
            extraction_response.append(response_text)
            extraction_model_response.append(response)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    all_responses = [
        (response, actual_label)
        for response, actual_label in zip(
            extraction_response, df["actual_label"].tolist()
        )
    ]
    batches = chunk_list(all_responses, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    args.max_tokens *= 2
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": qa_evaluate_answer(response, actual_label)}]
            for response, actual_label in batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for _ in range(len(batch)):
                evaluation_response.append(None)
                evaluation_model_response.append(str(e))
            continue

        for response in batch_responses:
            evaluation_model_response.append(response)
            try:
                response_text = response.choices[0].message.content.lower()  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_text = None
                answers.append(False)
                continue
            evaluation_response.append(response_text)
            find_correct = response_text.find("correct")  # type: ignore
            find_wrong = response_text.find("wrong")  # type: ignore
            answers.append(
                find_correct != -1 and (find_wrong == -1 or find_correct < find_wrong)
            )

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    # Add new columns to the DataFrame
    df["extraction_model_response"] = extraction_model_response
    df["extraction_response"] = extraction_response
    df["evaluation_model_response"] = evaluation_model_response
    df["evaluation_response"] = evaluation_response
    df["final_answer"] = answers

    # Compute Accuracy
    accuracy = len([a for a in answers if a]) / len(answers) if answers else 0.0
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy"],
            "Value": [accuracy],
        }
    )

    success_rate = df["extraction_response"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df, metrics_df
