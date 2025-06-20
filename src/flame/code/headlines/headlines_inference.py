from datetime import date

import pandas as pd
from tqdm import tqdm

from flame.code.prompts import PromptFormat, get_prompt
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.dataset_utils import safe_load_dataset
from flame.utils.logging_utils import get_component_logger

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "headlines")


def headlines_inference(args):
    today = date.today()
    logger.info(f"Starting Headlines inference on {today}")

    # Load the Headlines dataset (test split with specific config)
    logger.info("Loading dataset...")
    dataset = safe_load_dataset(
        "gtfintechlab/Headlines", name="5768", trust_remote_code=True
    )

    # Initialize lists to store news, model responses, labels, and actual labels
    news = []
    llm_responses = []
    complete_responses = []
    actual_labels = []  # List to store actual labels

    if args.prompt_format == "fewshot":
        headlines_prompt = get_prompt("headlines", PromptFormat.FEW_SHOT)
    else:
        headlines_prompt = get_prompt("headlines", PromptFormat.ZERO_SHOT)
    if headlines_prompt is None:
        raise RuntimeError("Headlines prompt not found in registry")

    test_data = dataset["test"]  # type: ignore
    all_sentences = [data["News"] for data in test_data]  # type: ignore
    all_actual_labels = [
        [
            data["PriceOrNot"],
            data["DirectionUp"],
            data["DirectionDown"],
            data["DirectionConstant"],
            data["PastPrice"],
            data["FuturePrice"],
            data["PastNews"],
        ]
        for data in test_data
    ]  # type: ignore

    batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch_content in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": headlines_prompt(sentence)}]
            for sentence in batch_content
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in batch_content:
                news.append(None)
                llm_responses.append(None)
                actual_labels.append(None)
                complete_responses.append(None)

        for sentence, response in zip(batch_content, batch_responses):
            news.append(sentence)
            try:
                response_text = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.error(f"Error processing sentence: {e}")
                response_text = None
            llm_responses.append(response_text)
            actual_labels.append(all_actual_labels[len(llm_responses) - 1])
            complete_responses.append(response)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "news": news,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
            "actual_labels": actual_labels,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
