import time
from datetime import date
import pandas as pd
from datasets import load_dataset

from litellm import batch_completion
from superflue.together_code.prompts import ectsum_prompt
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Setup logger for ectsum inference
logger = setup_logger(
    name="ectsum_inference", log_file=LOG_DIR / "ectsum_inference.log", level=LOG_LEVEL
)

def chunk_list(lst, chunk_size):
    """Split a list into chunks of the specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def process_batch_with_retry(args, messages_batch, batch_idx, total_batches):
    """Process a batch with retry mechanism."""
    try:
        batch_responses = batch_completion(
            model=args.model,
            messages=messages_batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_retries=3,
            repetition_penalty=args.repetition_penalty,
            stop=tokens(args.model),
        )
        logger.info(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses
    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {e}")
        raise

def ectsum_inference(args):
    today = date.today()
    logger.info(f"Starting ECTSum inference on {today}")

    # Load the ECTSum dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/ECTSum", trust_remote_code=True)
    
    results_path = (
        RESULTS_DIR
        / "ectsum"
        / f"ectsum_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract documents and actual labels
    documents = [row["context"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["response"] for row in dataset["test"]]  # type: ignore

    llm_responses = []
    complete_responses = []

    batch_size = 10
    total_batches = len(documents) // batch_size + int(len(documents) % batch_size > 0)
    logger.info(f"Processing {len(documents)} documents in {total_batches} batches.")

    document_batches = chunk_list(documents, batch_size)
    label_batches = chunk_list(actual_labels, batch_size)

    for batch_idx, (document_batch, label_batch) in enumerate(zip(document_batches, label_batches)):
        # Create message batches for the documents
        messages_batch = [
            [{"role": "user", "content": ectsum_prompt(document)}]
            for document in document_batch
        ]

        try:
            # Process the current batch
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)

            for response, actual_label in zip(batch_responses, label_batch):
                try:
                    response_text = response.choices[0].message.content.strip()  # type: ignore
                    llm_responses.append(response_text)
                    complete_responses.append(response)
                except (KeyError, IndexError, AttributeError) as e:
                    logger.error(f"Error extracting response for document in batch {batch_idx + 1}: {e}")
                    llm_responses.append("error")
                    complete_responses.append(None)

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            llm_responses.extend(["error"] * len(document_batch))
            complete_responses.extend([None] * len(document_batch))
            continue

    # Create the final DataFrame after processing all batches
    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    logger.info(f"Inference completed. Saving results to {results_path}.")
    df.to_csv(results_path, index=False)

    return df
