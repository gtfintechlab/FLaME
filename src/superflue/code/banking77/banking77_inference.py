import time
from datetime import date
import pandas as pd
from datasets import load_dataset

from superflue.code.inference_prompts import banking77_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
import litellm

from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

logger = setup_logger(
    name="banking77_inference", log_file=LOG_DIR / "banking77_inference.log", level=LOG_LEVEL
)


def banking77_inference(args):
    dataset = load_dataset("gtfintechlab/banking77", trust_remote_code=True)
    test_data = dataset["test"] # type: ignore
    all_documents = [data["text"] for data in test_data] # type: ignore
    all_actual_labels = [data["label"] for data in test_data] # type: ignore
    
    batch_size = args.batch_size
    total_batches = len(all_documents) // batch_size + int(len(all_documents) % batch_size > 0)
    logger.info(f"Processing {len(all_documents)} documents in {total_batches} batches.")
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
            [{"role": "user", "content": banking77_prompt(sentence)}]
            for sentence in sentence_batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
            
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add None values for failed batch
            for _ in sentence_batch:
                documents.append(None)
                complete_responses.append(None)
                llm_responses.append(None)
                actual_labels.append(None)
            continue
    
        # Process responses
        for sentence, response in zip(sentence_batch, batch_responses):
            documents.append(sentence)
            try: 
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            complete_responses.append(response)
            llm_responses.append(response_label)
            actual_labels.append(all_actual_labels[len(llm_responses) - 1])
            
        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = df["llm_responses"].notnull().sum() / len(df) * 100
    logger.info(f"Success rate: {success_rate}")
    
    return df
