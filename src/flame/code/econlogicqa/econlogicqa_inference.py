"""
NOTE: This task is not included in the current release.
EconLogicQA was not used in the camera-ready version of the paper
and will be implemented in a future release.
"""

from datetime import date

import pandas as pd
from tqdm import tqdm

from flame.code.prompts import PromptFormat, get_prompt
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.dataset_utils import safe_load_dataset
from flame.utils.logging_utils import get_component_logger

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "econlogicqa")


def econlogicqa_inference(args):
    today = date.today()
    logger.info(f"Starting EconLogicQA inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = safe_load_dataset("gtfintechlab/econlogicqa", trust_remote_code=True)[
        "test"
    ]

    # Prepare for batch processing
    instances = []
    for i in range(len(dataset)):
        row = dataset[i]
        question = row["Question"]
        event_a = row["A"]
        event_b = row["B"]
        event_c = row["C"]
        event_d = row["D"]
        instances.append((row, question, event_a, event_b, event_c, event_d))

    logger.info(f"Found {len(instances)} instances for processing")

    # Set up prompt format
    if args.prompt_format == "fewshot":
        econlogicqa_prompt = get_prompt("econlogicqa", PromptFormat.FEW_SHOT)
    else:
        econlogicqa_prompt = get_prompt("econlogicqa", PromptFormat.ZERO_SHOT)

    if econlogicqa_prompt is None:
        raise RuntimeError("EconLogicQA prompt not found in registry")

    # Create batches for processing
    batches = chunk_list(instances, args.batch_size)
    total_batches = len(batches)

    responses_ordered_importance = []

    pbar = tqdm(batches, desc="Processing EconLogicQA entries")
    for batch_idx, instance_batch in enumerate(pbar):
        # Prepare messages for the batch
        messages_batch = [
            [
                {
                    "role": "user",
                    "content": econlogicqa_prompt(
                        question, event_a, event_b, event_c, event_d
                    ),
                }
            ]
            for _, question, event_a, event_b, event_c, event_d in instance_batch
        ]

        try:
            # Process batch with retry mechanism
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            # Process responses
            for (row_data, _, _, _, _, _), response in zip(
                instance_batch, batch_responses
            ):
                row = row_data.copy()

                try:
                    llm_response = response.choices[0].message.content
                    ordered_response = (
                        llm_response.splitlines()[0]
                        if llm_response and "\n" in llm_response
                        else llm_response
                    )
                except Exception as e:
                    logger.error(
                        f"Error in response parsing: {str(e)}\nResponse: {response}"
                    )
                    ordered_response = "Error"
                    llm_response = f"Error: {str(e)}"

                row["llm_responses"] = ordered_response
                row["llm_complete_responses"] = llm_response
                responses_ordered_importance.append(row)

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add error values for the entire failed batch
            for row_data, _, _, _, _, _ in instance_batch:
                row = row_data.copy()
                row["llm_responses"] = "Error"
                row["llm_complete_responses"] = f"Error: {str(e)}"
                responses_ordered_importance.append(row)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    output_df = pd.DataFrame(responses_ordered_importance)

    # Calculate success rate
    success_rate = (
        sum(1 for r in output_df["llm_responses"] if r != "Error") / len(output_df)
    ) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return output_df
