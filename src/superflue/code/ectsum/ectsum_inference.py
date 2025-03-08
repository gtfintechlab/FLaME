import time
from datetime import date
import pandas as pd
from datasets import load_dataset

from litellm import completion 
from superflue.code.inference_prompts import ectsum_prompt
# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Setup logger for ectsum inference
logger = setup_logger(
    name="ectsum_inference", log_file=LOG_DIR / "ectsum_inference.log", level=LOG_LEVEL
)
import litellm
litellm.drop_params = True

def ectsum_inference(args):

    today = date.today()
    logger.info(f"Starting ECTSum inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/ECTSum", trust_remote_code=True)
    
    results_path = (
            RESULTS_DIR
            / "ectsum"
            / f"ectsum_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    documents = [row["context"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["response"] for row in dataset["test"]]  # type: ignore
    llm_responses = []
    complete_responses = []
    
    batch_size = 1
    total_batches = len(documents) // batch_size + int(len(documents) % batch_size > 0)

    logger.info(f"Processing {len(documents)} documents in {total_batches} batches.")

    document_batches = chunk_list(documents, batch_size)
    label_batches = chunk_list(actual_labels, batch_size)
    
    for batch_idx, document_batch in enumerate(document_batches):
        # Prepare messages for batch processing
        messages_batch = [
            [{"role": "user", "content": ectsum_prompt(doc)}] for doc in document_batch
        ]

        try:
            # Use batch processing with retry mechanism
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)

            for response in batch_responses:
                try:
                    response_text = response.choices[0].message.content.strip()  # type: ignore
                    llm_responses.append(response_text)
                    complete_responses.append(response)
                except (KeyError, IndexError, AttributeError) as e:
                    logger.error(f"Error extracting response: {e}")
                    llm_responses.append("error")
                    complete_responses.append(None)
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            llm_responses.extend(["error"] * len(document_batch))
            complete_responses.extend([None] * len(document_batch))
            continue
        
    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    logger.info(f"Inference completed. Returning DataFrame with {len(df)} rows.")

    return df
