import pandas as pd
from datasets import load_dataset
from flame.code.inference_prompts import headlines_prompt
from flame.utils.batch_utils import process_batch_with_retry, chunk_list
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm

# Setup logger for Headlines inference
logger = setup_logger(
    name="headlines_inference",
    log_file=LOG_DIR / "headlines_inference.log",
    level=LOG_LEVEL,
)


def headlines_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/Headlines", "5768", trust_remote_code=True)

    llm_responses = []
    complete_responses = []

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
                llm_responses.append(None)
                complete_responses.append(None)
            continue

        for response in batch_responses:
            try:
                response_text = response.choices[0].message.content.strip()  # type: ignore
            except Exception as e:
                logger.error(f"Error processing sentence: {e}")
                response_text = None
            llm_responses.append(response_text)
            complete_responses.append(response)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "news": all_sentences,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
            "actual_labels": all_actual_labels,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
