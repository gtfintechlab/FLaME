import pandas as pd
from datasets import load_dataset
from superflue.code.inference_prompts import finer_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm

logger = setup_logger(
    name="finer_inference", log_file=LOG_DIR / "finer_inference.log", level=LOG_LEVEL
)


def finer_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/finer-ord-bio", trust_remote_code=True)

    sentences = [row["tokens"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["tags"] for row in dataset["test"]]  # type: ignore

    llm_responses = []
    complete_responses = []

    batches = chunk_list(sentences, args.batch_size)
    total_batches = len(batches)
    logger.info(f"Processing {len(sentences)} sentences in {total_batches} batches.")

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": finer_prompt(sentence)}] for sentence in batch
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
            try:
                response_label = response.choices[0].message.content.strip()  # type: ignore
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting response: {e}")
                response_label = "Error"
            llm_responses.append(response_label)
            complete_responses.append(response)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

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
