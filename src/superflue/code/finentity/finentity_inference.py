import pandas as pd
from datasets import load_dataset
from superflue.code.inference_prompts import finentity_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR, LOG_LEVEL
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from tqdm import tqdm

logger = setup_logger(
    name="finentity_inference",
    log_file=LOG_DIR / "finentity_inference.log",
    level=LOG_LEVEL,
)


def finentity_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/finentity", "5768", trust_remote_code=True)

    # Extract sentences and actual labels
    sentences = [row["content"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["annotations"] for row in dataset["test"]]  # type: ignore

    llm_responses = []
    complete_responses = []

    batch_size = args.batch_size
    total_batches = len(sentences) // batch_size + int(len(sentences) % batch_size > 0)
    logger.info(f"Processing {len(sentences)} sentences in {total_batches} batches.")

    sentence_batches = chunk_list(sentences, batch_size)

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": finentity_prompt(sentence)}]
            for sentence in batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            for _ in batch:
                llm_responses.append("Error")
                complete_responses.append(None)
            continue

        for response in batch_responses:
            complete_responses.append(response)
            try:
                response_label = response.choices[0].message.content.strip()  # type: ignore
                llm_responses.append(response_label)
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting response: {e}")
                llm_responses.append("Error")

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    # Create the final DataFrame
    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = df["llm_responses"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")

    return df
