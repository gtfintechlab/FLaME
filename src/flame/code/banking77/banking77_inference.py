from datetime import date

import pandas as pd
from tqdm import tqdm

from flame.code.prompts import PromptFormat, get_prompt
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.dataset_utils import safe_load_dataset
from flame.utils.logging_utils import get_component_logger

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "banking77")


def banking77_inference(args):
    today = date.today()
    logger.info(f"Starting Banking77 inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = safe_load_dataset("gtfintechlab/banking77", trust_remote_code=True)
    test_data = dataset["test"]  # type: ignore
    all_documents = [data["text"] for data in test_data]  # type: ignore
    all_actual_labels = [data["label"] for data in test_data]  # type: ignore

    sentence_batches = chunk_list(all_documents, args.batch_size)
    total_batches = len(sentence_batches)

    documents = []
    llm_responses = []
    actual_labels = []
    complete_responses = []
    if args.prompt_format == "fewshot":
        banking77_prompt = get_prompt("banking77", PromptFormat.FEW_SHOT)
    else:
        banking77_prompt = get_prompt("banking77", PromptFormat.ZERO_SHOT)
    if banking77_prompt is None:
        raise RuntimeError("Banking77 prompt not found in registry")

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": banking77_prompt(sentence)}]
            for sentence in sentence_batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add None values for failed batch
            for _ in sentence_batch:
                documents.append(None)
                complete_responses.append(None)
                llm_responses.append(None)
                actual_labels.append(None)
            continue

        # Process responses
        for sentence, response in zip(sentence_batch, batch_responses):
            documents.append(sentence)
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            complete_responses.append(response)
            llm_responses.append(response_label)
            actual_labels.append(all_actual_labels[len(llm_responses) - 1])

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Calculate success rate
    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
