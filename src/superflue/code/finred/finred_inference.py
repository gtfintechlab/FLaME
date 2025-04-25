from pathlib import Path
import time
from datetime import date
import pandas as pd
from datasets import load_dataset

from litellm import completion 
import litellm
from typing import Dict, Any, List, Optional, Tuple
from superflue.code.prompts_zeroshot import finred_prompt
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from tqdm import tqdm

# Setup logger for FinRED inference
logger = setup_logger(
    name="finred_inference", log_file=LOG_DIR / "finred_inference.log", level=LOG_LEVEL
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
            # top_k=args.top_k if args.top_k else None,
            top_p=args.top_p,
            # repetition_penalty=args.repetition_penalty,
            num_retries=3  # Using litellm's retry mechanism
        )
        logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses
            
    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise

def finred_inference(args):
    today = date.today()
    logger.info(f"Starting FinRED inference on {today}")

    # Load the FinRED dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/FinRed", trust_remote_code=True)

    # Initialize lists to store sentences, actual labels, model responses, and complete responses
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []
    entities_list = []  # To store entity pairs

    test_data = dataset["test"]  # type: ignore
    all_inputs = [(data["sentence"], data["entities"]) for data in test_data]  # type: ignore
    all_inputs = [(input[0], entity_pair) for input in all_inputs for entity_pair in input[1]]
    all_actual_labels = [data["relations"] for data in test_data]  # type: ignore
    all_actual_labels = [label for labels in all_actual_labels for label in labels]

    batches = chunk_list(all_inputs, args.batch_size)
    total_batches = len(batches)

    logger.info(f"Starting inference on FinRED with model {args.model}...")

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": finred_prompt(input[0], input[1][0], input[1][1])}]
            for input in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add None values for failed batch
            for _ in batch:
                sentences.append(None)
                entities_list.append(None)
                complete_responses.append(None)
                llm_responses.append(None)
                actual_labels.append(None)
            continue

        # Process responses
        for (sentence, entity_pair), response in zip(batch, batch_responses):
            sentences.append(sentence)
            entities_list.append(entity_pair)
            complete_responses.append(response)
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            actual_labels.append(all_actual_labels[len(llm_responses) - 1])
            
        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "sentence": sentences,
            "entity_pairs": entities_list,
            "actual_labels": actual_labels,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df['llm_responses'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df