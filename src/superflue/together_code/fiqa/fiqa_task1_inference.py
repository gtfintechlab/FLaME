import time
import pandas as pd
from datetime import date
from datasets import load_dataset
from litellm import completion 
from superflue.together_code.prompts import fiqa_task1_prompt
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
import litellm
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

# Set up logger
logger = setup_logger(
    name="fiqa_task1_inference",
    log_file=LOG_DIR / "fiqa_task1_inference.log",
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

def fiqa_task1_inference(args):
    # Load dataset and initialize storage for results
    dataset = load_dataset("gtfintechlab/FiQA_Task1", split="test", trust_remote_code=True)

    test_data = dataset["test"] # type: ignore
    all_texts = [f"Sentence: {data['sentence']}. Snippets: {data['snippets']}. Target aspect: {data['target']}" for data in test_data] # type: ignore
    all_targets = [data["target"] for data in test_data] # type: ignore
    all_sentiments = [data["sentiment_score"] for data in test_data] # type: ignore

    sentence_batches = chunk_list(all_texts, args.batch_size)
    total_batches = len(sentence_batches)
    context = []
    llm_responses = []
    actual_targets = []
    actual_sentiments = []
    complete_responses = []

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": fiqa_task1_prompt(sentence)}]
            for sentence in sentence_batch
        ]
        try:
            # Process batch with retry mechanism
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)
            for sentence, response in zip(sentence_batch, batch_responses):
                response_label = response.choices[0].message.content
                llm_responses.append(response_label)
                complete_responses.append(response)
                context.append(sentence)
                actual_targets.append(all_targets[len(llm_responses) - 1])
                actual_sentiments.append(all_sentiments[len(llm_responses) - 1])
            pbar.set_description(f"Completed batch {batch_idx + 1}/{total_batches}")

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(sentence_batch)):
                llm_responses.append(None)
                complete_responses.append(None)
                context.append(None)
                actual_targets.append(None)
                actual_sentiments.append(None)
    
    # Create DataFrame with results
    df = pd.DataFrame(
        {
            "context": context,
            "llm_responses": llm_responses,
            "actual_target": actual_targets,
            "actual_sentiment": actual_sentiments,
            "complete_responses": complete_responses,
        }
    )

    return df
