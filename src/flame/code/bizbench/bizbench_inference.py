from datetime import date

import pandas as pd
from tqdm import tqdm

from flame.code.prompts import PromptFormat, get_prompt
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.dataset_utils import safe_load_dataset
from flame.utils.logging_utils import get_component_logger

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "bizbench")


def bizbench_inference(args):
    today = date.today()
    logger.info(f"Starting BizBench inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = safe_load_dataset("kensho/bizbench", trust_remote_code=True)

    # Initialize lists to store actual labels and model responses
    X_question = []
    X_context = []
    y_answer = []
    llm_responses = []
    complete_responses = []

    # Extract test instances
    test_data = dataset["test"]  # type: ignore
    instances = []

    # Preprocess to filter valid instances
    for i in range(len(test_data)):
        instance = test_data[i]
        question = instance["question"]
        answer = instance["answer"]
        context = instance["context"]

        # Skip instances with no context
        if not context:
            continue

        instances.append((question, answer, context))

    logger.info(f"Found {len(instances)} valid instances for processing")

    # Set up prompt
    if args.prompt_format == "fewshot":
        bizbench_prompt = get_prompt("bizbench", PromptFormat.FEW_SHOT)
    else:
        bizbench_prompt = get_prompt("bizbench", PromptFormat.ZERO_SHOT)
    if bizbench_prompt is None:
        raise RuntimeError("BizBench prompt not found in registry")

    # Create batches for processing
    batches = chunk_list(instances, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing BizBench entries")
    for batch_idx, instance_batch in enumerate(pbar):
        # Prepare messages for the batch
        messages_batch = [
            [{"role": "user", "content": bizbench_prompt(question, context)}]
            for question, _, context in instance_batch
        ]

        try:
            # Process batch with retry mechanism
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            # Process responses
            for (question, answer, context), response in zip(
                instance_batch, batch_responses
            ):
                X_question.append(question)
                X_context.append(context)
                y_answer.append(answer)
                complete_responses.append(response)

                try:
                    response_label = response.choices[0].message.content
                    llm_responses.append(response_label)
                except Exception as e:
                    logger.error(
                        f"Error in response parsing: {str(e)}\nResponse: {response}"
                    )
                    llm_responses.append(None)

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add None values for the entire failed batch
            for _ in instance_batch:
                X_question.append(None)
                X_context.append(None)
                y_answer.append(None)
                complete_responses.append(None)
                llm_responses.append(None)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "X_question": X_question,
            "X_context": X_context,
            "y_answer": y_answer,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    # Calculate success rate
    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
