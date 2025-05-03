import pandas as pd
from datasets import load_dataset
from flame.code.inference_prompts import causal_classification_prompt
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_LEVEL, LOG_DIR
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from tqdm import tqdm

logger = setup_logger(
    name="causal_classification_inference",
    log_file=LOG_DIR / "causal_classification_inference.log",
    level=LOG_LEVEL,
)


def causal_classification_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")

    dataset = load_dataset("gtfintechlab/CausalClassification", trust_remote_code=True)

    texts = [row["text"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["label"] for row in dataset["test"]]  # type: ignore
    llm_responses = []
    complete_responses = []

    batch_size = args.batch_size
    total_batches = len(texts) // batch_size + int(len(texts) % batch_size > 0)
    logger.info(f"Processing {len(texts)} texts in {total_batches} batches.")

    text_batches = chunk_list(texts, batch_size)

    pbar = tqdm(text_batches, desc="Processing batches")
    for batch_idx, text_batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": causal_classification_prompt(text)}]
            for text in text_batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            llm_responses.extend(["error"] * len(text_batch))
            complete_responses.extend([None] * len(text_batch))
            continue

        for response in batch_responses:
            complete_responses.append(response)
            try:
                response_label = response.choices[0].message.content.strip()  # type: ignore
                llm_responses.append(response_label)
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting response: {e}")
                llm_responses.append("error")

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df = pd.DataFrame(
        {
            "texts": texts,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
