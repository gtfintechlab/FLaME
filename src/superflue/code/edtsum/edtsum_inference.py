import time
import pandas as pd
from datasets import load_dataset
from litellm import completion 
from superflue.code.prompts import edtsum_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.code.tokens import tokens
from superflue.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm
import litellm
from typing import Dict, Any, List, Optional, Tuple
from litellm.utils import trim_messages, get_max_tokens

logger = setup_logger(
    name="edtsum_inference", log_file=LOG_DIR / "edtsum_inference.log", level=LOG_LEVEL
)

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def process_batch_with_retry(args, messages_batch, batch_idx, total_batches):
    """Process a batch with litellm's retry mechanism."""
    try:
        # Using litellm's built-in retry mechanism
        batch_responses = litellm.batch_completion(
            model=args.model,
            messages=messages_batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k else None,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_retries=3  # Using litellm's retry mechanism
        )
        logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses
            
    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise

def edtsum_inference(args):
    # today = date.today()

    dataset = load_dataset("gtfintechlab/EDTSum", trust_remote_code=True)

    test_data = dataset["test"] # type: ignore
    all_documents = [data["text"] for data in test_data] # type: ignore
    all_actual_labels = [data["answer"] for data in test_data] # type: ignore

    sentence_batches = chunk_list(all_documents, args.batch_size)
    total_batches = len(sentence_batches)

    documents = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            trim_messages([{"role": "user", "content": edtsum_prompt(document)}], max_tokens=get_max_tokens(args.model) - args.max_tokens - 1)
            for document in sentence_batch
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            for document, response in zip(sentence_batch, batch_responses):
                documents.append(document)
                response_label = response.choices[0].message.content # type: ignore
                llm_responses.append(response_label)
                complete_responses.append(response)
                actual_labels.append(all_actual_labels[len(llm_responses) - 1])

            pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(sentence_batch)):
                documents.append(None)
                llm_responses.append(None)
                complete_responses.append(None)
                actual_labels.append(None)
            continue

    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    return df
