from datetime import date
import pandas as pd
from datasets import load_dataset

from superflue.code.prompts_zeroshot import headlines_zeroshot_prompt
from superflue.code.prompts_fewshot import headlines_fewshot_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR, LOG_LEVEL
import litellm
from typing import Any, List
from tqdm import tqdm

# Setup logger for Headlines inference
logger = setup_logger(
    name="headlines_inference",
    log_file=LOG_DIR / "headlines_inference.log",
    level=LOG_LEVEL,
)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_batch_with_retry(args, messages_batch, batch_idx, total_batches):
    """Process a batch with litellm's retry mechanism."""
    try:
        # Using litellm's built-in retry mechanism
        batch_responses = litellm.batch_completion(
            model=args.model,
            messages=messages_batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            # top_k=args.top_k if args.top_k else None,
            top_p=args.top_p,
            # repetition_penalty=args.repetition_penalty,
            num_retries=3,  # Using litellm's retry mechanism
        )
        logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses

    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise


def headlines_inference(args):
    today = date.today()
    logger.info(f"Starting Headlines inference on {today}")

    # Load the Headlines dataset (test split with specific config)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/Headlines", "5768", trust_remote_code=True)

    # Initialize lists to store news, model responses, labels, and actual labels
    news = []
    llm_responses = []
    complete_responses = []
    actual_labels = []  # List to store actual labels

    if args.prompt_format == "fewshot":
        headlines_prompt = headlines_fewshot_prompt
    elif args.prompt_format == "zeroshot":
        headlines_prompt = headlines_zeroshot_prompt

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
