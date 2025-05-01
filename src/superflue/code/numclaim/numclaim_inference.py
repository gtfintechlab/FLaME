import time
import litellm
from datetime import date
import pandas as pd
from datasets import load_dataset

from superflue.code.prompts import numclaim_prompt

# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry

# Setup logger for Numclaim inference
logger = setup_logger(
    name="numclaim_inference",
    log_file=LOG_DIR / "numclaim_inference.log",
    level=LOG_LEVEL,
)

litellm.drop_params = True


# litellm.set_verbose = True
# litellm._turn_on_debug()
def numclaim_inference(args):
    today = date.today()
    logger.info(f"Starting Numclaim inference on {today}")

    # Load the Numclaim dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/Numclaim", trust_remote_code=True)

    results_path = (
        RESULTS_DIR
        / "numclaim"
        / f"numclaim_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize lists to store sentences, actual labels, model responses, and complete responses
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on Numclaim with model {args.model}...")

    sentences = [row["context"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["response"] for row in dataset["test"]]  # type: ignore
    batch_size = args.batch_size
    total_batches = len(sentences) // batch_size + int(len(sentences) % batch_size > 0)
    logger.info(f"Processing {len(sentences)} rows in {total_batches} batches.")

    # Create batches
    sentence_batches = chunk_list(sentences, batch_size)
    response_batches = chunk_list(actual_labels, batch_size)

    for batch_idx, (sentence_batch, response_batch) in enumerate(
        zip(sentence_batches, response_batches)
    ):
        # Create prompt messages for the batch
        messages_batch = [
            [{"role": "user", "content": numclaim_prompt(sentence)}]  # type: ignore
            for sentence in zip(sentence_batch)
        ]

        try:
            # Process the batch
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            for response in batch_responses:
                try:
                    llm_response = response.choices[0].message.content.strip()  # type: ignore
                    llm_responses.append(llm_response)
                    complete_responses.append(response)
                except (KeyError, IndexError, AttributeError) as e:
                    logger.error(f"Error extracting response: {e}")
                    llm_responses.append("error")
                    complete_responses.append(None)
                finally:
                    time.sleep(1)  # Sleep for 1 second after each response

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            llm_responses.extend(["error"] * len(sentence_batch))
            complete_responses.extend([None] * len(sentence_batch))
            continue

    # Create the final DataFrame
    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )
    results_path = (
        RESULTS_DIR
        / "numclaim"
        / f"numclaim_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    # Save the results to a CSV file
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
