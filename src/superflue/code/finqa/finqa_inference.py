import time

import pandas as pd
from datasets import load_dataset
from litellm import completion 
from datetime import date
from superflue.code.prompts import finqa_prompt
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
import litellm
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

# TODO: (Glenn) Is FinQA saving results to a file properly?

logger = setup_logger(
    name="finqa_inference", log_file=LOG_DIR / "finqa_inference.log", level=LOG_LEVEL
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

def finqa_inference(args):
    dataset = load_dataset("gtfintechlab/finqa", trust_remote_code=True)
    test_data = dataset["test"]  # type: ignore
    all_texts = [f"{' '.join(data['pre_text'])} {' '.join(data['post_text'])} {' '.join([' '.join(row) for row in data['table_ori']])} {data['question']}" for data in test_data]  # type: ignore
    all_actual_labels = [data["answer"] for data in test_data]  # type: ignore
    text_batches = chunk_list(all_texts, args.batch_size)
    total_batches = len(text_batches)

    context = []
    llm_responses = []
    actual_labels = []
    complete_responses = []
    
    pbar = tqdm(text_batches, desc="Processing batches")
    for batch_idx, text_batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": finqa_prompt(sentence)}]
            for sentence in text_batch
        ]
        try:
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in text_batch:
                context.append(None)
                llm_responses.append(None)
                complete_responses.append(None)
                actual_labels.append(None)
        
        for text, response in zip(text_batch, batch_responses):
            context.append(text)
            try:
                response_label = response.choices[0].message.content  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            complete_responses.append(response)
            actual_labels.append(all_actual_labels[len(llm_responses) - 1])
            
        pbar.set_description(f"Completed batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "context": context,
            "response": llm_responses,
            "actual_label": actual_labels,
            "complete_responses": complete_responses,
        }
    )
    
    return df
