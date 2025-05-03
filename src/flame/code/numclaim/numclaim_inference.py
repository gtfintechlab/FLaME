import pandas as pd
from datasets import load_dataset
from flame.code.inference_prompts import numclaim_prompt
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from tqdm import tqdm

logger = setup_logger(
    name="numclaim_inference",
    log_file=LOG_DIR / "numclaim_inference.log",
    level=LOG_LEVEL,
)


def numclaim_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/Numclaim", trust_remote_code=True)

    llm_responses = []
    complete_responses = []

    logger.info(f"Starting inference on Numclaim with model {args.model}...")

    sentences = [row["context"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["response"] for row in dataset["test"]]  # type: ignore

    batches = chunk_list(sentences, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch_content in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": numclaim_prompt(sentence)}]  # type: ignore
            for sentence in batch_content
        ]

        try:
            # Process the batch
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            for _ in batch_content:
                llm_responses.append(None)
                complete_responses.append(None)
            continue

        for response in batch_responses:
            try:
                llm_response = response.choices[0].message.content.strip()  # type: ignore
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting response: {e}")
                llm_response = None

            complete_responses.append(response)
            llm_responses.append(llm_response)

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
