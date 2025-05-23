from datetime import date

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import get_component_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "convfinqa")


def convfinqa_inference(args):
    today = date.today()
    logger.info(f"Starting ConvFinQA inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/convfinqa", trust_remote_code=True)

    # Prepare data for batch processing
    context = []
    llm_responses = []
    actual_labels = []
    complete_responses = []
    instances = []

    # Set up prompt format
    if args.prompt_format == "fewshot":
        convfinqa_prompt = get_prompt("convfinqa", PromptFormat.FEW_SHOT)
    else:
        convfinqa_prompt = get_prompt("convfinqa", PromptFormat.ZERO_SHOT)
    if convfinqa_prompt is None:
        raise RuntimeError("ConvFinQA prompt not found in registry")

    # Pre-process all entries to prepare for batching
    logger.info("Preprocessing dataset entries...")
    for entry in dataset["train"]:  # type: ignore
        pre_text = " ".join(entry["pre_text"])  # type: ignore
        post_text = " ".join(entry["post_text"])  # type: ignore
        table_text = " ".join([" ".join(map(str, row)) for row in entry["table_ori"]])  # type: ignore
        question_0 = str(entry["question_0"]) if entry["question_0"] is not None else ""  # type: ignore
        question_1 = str(entry["question_1"]) if entry["question_1"] is not None else ""  # type: ignore
        answer_0 = str(entry["answer_0"]) if entry["answer_0"] is not None else ""  # type: ignore
        # answer_1 = str(entry["answer_1"]) if entry["answer_1"] is not None else ""  # type: ignore
        combined_text = f"{pre_text} {post_text} {table_text} Question 0: {question_0} Answer: {answer_0}. Now answer the following question: {question_1}"
        actual_label = entry["answer_1"]  # type: ignore

        instances.append((combined_text, actual_label))

    logger.info(f"Found {len(instances)} instances for processing")

    # Create batches for processing
    batches = chunk_list(instances, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing ConvFinQA entries")
    for batch_idx, instance_batch in enumerate(pbar):
        # Prepare messages for the batch
        messages_batch = [
            [{"role": "user", "content": convfinqa_prompt(combined_text)}]
            for combined_text, _ in instance_batch
        ]

        try:
            # Process batch with retry mechanism
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            # Process responses
            for (combined_text, actual_label), response in zip(
                instance_batch, batch_responses
            ):
                context.append(combined_text)
                actual_labels.append(actual_label)
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
                context.append(None)
                actual_labels.append(None)
                complete_responses.append(None)
                llm_responses.append(None)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "context": context,
            "response": llm_responses,
            "actual_label": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Calculate success rate
    success_rate = (df["response"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
