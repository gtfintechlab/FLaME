from datetime import date
import pandas as pd
from datasets import load_dataset
from litellm import batch_completion
from superflue.code.inference_prompts import causal_classification_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_LEVEL, LOG_DIR, RESULTS_DIR
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
import time

logger = setup_logger(
    name="causal_classification_inference",
    log_file=LOG_DIR / "causal_classification_inference.log",
    level=LOG_LEVEL,
)

import litellm
litellm.drop_params = True

def causal_classification_inference(args):
    today = date.today()
    logger.info(f"Starting Causal Classification inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/CausalClassification", trust_remote_code=True)

    texts = [row["text"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["label"] for row in dataset["test"]]  # type: ignore
    llm_responses = []
    complete_responses = []

    batch_size = args.batch_size
    total_batches = len(texts) // batch_size + int(len(texts) % batch_size > 0)
    logger.info(f"Processing {len(texts)} texts in {total_batches} batches.")

    text_batches = chunk_list(texts, batch_size)
    label_batches = chunk_list(actual_labels, batch_size)

    for batch_idx, text_batch in enumerate(text_batches):
        messages_batch = [
            [{"role": "user", "content": causal_classification_prompt(text)}]
            for text in text_batch
        ]

        try:
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)
            # time.sleep(1)

            for response in batch_responses:
                try:
                    response_label = response.choices[0].message.content.strip()  # type: ignore
                    llm_responses.append(response_label)
                    complete_responses.append(response)
                except (KeyError, IndexError, AttributeError) as e:
                    logger.error(f"Error extracting response: {e}")
                    llm_responses.append("error")
                    complete_responses.append(None)

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            llm_responses.extend(["error"] * len(text_batch))
            complete_responses.extend([None] * len(text_batch))
            continue

    df = pd.DataFrame(
        {
            "texts": texts,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    results_path = (
        RESULTS_DIR
        / "causal_classification"
        / f"causal_classification_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
