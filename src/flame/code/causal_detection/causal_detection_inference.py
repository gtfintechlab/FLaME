from datetime import date

import pandas as pd
from tqdm import tqdm

from flame.code.prompts import PromptFormat, get_prompt
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.dataset_utils import safe_load_dataset
from flame.utils.logging_utils import get_component_logger

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "causal_detection")


def causal_detection_inference(args):
    today = date.today()
    logger.info(f"Starting Causal Detection inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = safe_load_dataset("gtfintechlab/CausalDetection", trust_remote_code=True)

    test_data = dataset["test"]  # type: ignore
    all_tokens = [data["tokens"] for data in test_data]  # type: ignore
    all_actual_tags = [data["tags"] for data in test_data]  # type: ignore

    logger.info(f"Found {len(all_tokens)} instances for processing")

    # Initialize lists to store tokens, actual tags, predicted tags, and complete responses
    tokens_list = []
    actual_tags = []
    llm_responses = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        causal_detection_prompt = get_prompt("causal_detection", PromptFormat.FEW_SHOT)
    else:
        causal_detection_prompt = get_prompt("causal_detection", PromptFormat.ZERO_SHOT)
    if causal_detection_prompt is None:
        raise RuntimeError("Causal Detection prompt not found in registry")

    batches = chunk_list(all_tokens, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing entries")
    for batch_idx, token_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": causal_detection_prompt(tokens)}]
            for tokens in token_batch
        ]

        try:
            # Process batch with retry mechanism
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add None values for failed batch
            for _ in token_batch:
                tokens_list.append(None)
                actual_tags.append(None)
                complete_responses.append(None)
                llm_responses.append(None)

        for token, response in zip(token_batch, batch_responses):
            complete_responses.append(response)
            try:
                response_label = response.choices[0].message.content
                response_tags = response_label.split()
            except Exception as e:
                logger.debug(f"Error in response: {str(e)}\nResponse: {response}")
                response_tags = None
            llm_responses.append(response_tags)
            tokens_list.append(token)
            actual_tags.append(all_actual_tags[len(llm_responses) - 1])

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "tokens": tokens_list,
            "actual_tags": actual_tags,
            "predicted_tags": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    # Calculate success metrics

    success_rate = (df["predicted_tags"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
