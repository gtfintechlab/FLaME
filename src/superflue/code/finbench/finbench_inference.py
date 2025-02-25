import time
from datetime import date

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from superflue.code.prompts_oldsuperflue import finbench_prompt
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

from litellm import completion 
import litellm
from typing import Dict, Any, List, Optional, Tuple

logger = setup_logger(
    name="finbench_inference",
    log_file=LOG_DIR / "finbench_inference.log",
    level=LOG_LEVEL,
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


def finbench_inference(args):
    today = date.today()
    logger.info(f"Starting FinBench inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/finbench", trust_remote_code=True)

    # Initialize lists to store actual labels and model responses
    X_profile_data = []
    y_data = []
    llm_responses = []
    complete_responses = []

    test_data = dataset["test"] # type: ignore
    all_profiles = [data["X_profile"] for data in test_data] # type: ignore
    all_actual_labels = [data["y"] for data in test_data] # type: ignore

    sentence_batches = chunk_list(all_profiles, args.batch_size)
    total_batches = len(sentence_batches)

    logger.info("Starting inference on dataset...")
    # start_t = time.time()

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": finbench_prompt(profile)}]
            for profile in sentence_batch
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
                X_profile_data.append(None)
                complete_responses.append(None)
                llm_responses.append(None)
                y_data.append(None)
            continue
    
        # Process responses
        for profile, response in zip(sentence_batch, batch_responses):
            X_profile_data.append(profile)
            complete_responses.append(response)
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            y_data.append(all_actual_labels[len(llm_responses) - 1])
        
        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "X_profile": all_profiles,
            "y": y_data,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df['llm_responses'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")
    
    return df
