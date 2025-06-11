from datetime import date

import pandas as pd
from tqdm import tqdm

from flame.code.prompts import PromptFormat, get_prompt
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.dataset_utils import safe_load_dataset
from flame.utils.logging_utils import get_component_logger

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "fiqa_task2")


def fiqa_task2_inference(args):
    today = date.today()
    logger.info(f"Starting FiQA Task 2 inference on {today}")

    # Load dataset and initialize lists for results
    logger.info("Loading dataset...")
    dataset = safe_load_dataset("gtfintechlab/FiQA_Task2", trust_remote_code=True)

    test_data = dataset["test"]  # type: ignore
    all_questions = [data["question"] for data in test_data]  # type: ignore
    all_answers = [data["answer"] for data in test_data]  # type: ignore

    question_batches = chunk_list(all_questions, args.batch_size)
    total_batches = len(question_batches)

    context = []
    llm_responses = []
    actual_answers = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        fiqa_task2_prompt = get_prompt("fiqa_task2", PromptFormat.FEW_SHOT)
    else:
        fiqa_task2_prompt = get_prompt("fiqa_task2", PromptFormat.ZERO_SHOT)
    if fiqa_task2_prompt is None:
        raise RuntimeError("FiQA Task2 prompt not found in registry")

    pbar = tqdm(question_batches, desc="Processing batches")
    for batch_idx, question_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": fiqa_task2_prompt(question)}]
            for question in question_batch
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(question_batch)):
                context.append(None)
                complete_responses.append(None)
                llm_responses.append(None)
                actual_answers.append(None)

        for question, response in zip(question_batch, batch_responses):
            context.append(question)
            complete_responses.append(response)
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            actual_answers.append(all_answers[len(llm_responses) - 1])
        pbar.set_description(f"Completed batch {batch_idx + 1}/{total_batches}")

    # Save results intermittently
    df = pd.DataFrame(
        {
            "question": context,
            "llm_responses": llm_responses,
            "actual_answers": actual_answers,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
