import time
from datetime import date
from pathlib import Path
import pandas as pd
from datasets import load_dataset
import together
from superflue.code.prompts_oldsuperflue import causal_detection_prompt
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from litellm import completion
import litellm
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

logger = setup_logger(
    name="cd_inference", log_file=LOG_DIR / "cd_inference.log", level=LOG_LEVEL
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

def casual_detection_inference(args):
    today = date.today()
    dataset = load_dataset("gtfintechlab/CausalDetection", trust_remote_code=True)

    test_data = dataset["test"]  # type: ignore
    all_tokens = [data["tokens"] for data in test_data]  # type: ignore
    all_actual_tags = [data["tags"] for data in test_data]  # type: ignore

    # Initialize lists to store tokens, actual tags, predicted tags, and complete responses
    tokens_list = []
    actual_tags = []
    llm_responses = []
    complete_responses = []

    batches = chunk_list(all_tokens, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing entries")
    for batch_idx, token_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": causal_detection_prompt(tokens)}]
            for tokens in token_batch
        ]

        try:
            # Process batch with retry mechanism
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add None values for failed batch
            for _ in token_batch:
                tokens_list.append(None)
                actual_tags.append(None)
                complete_responses.append(None)
                llm_responses.append(None)
        
        for token, response in zip(token_batch, batch_responses):
            complete_responses.append(response)
            try: 
                response_label = response.choices[0].message.content
                response_tags = response_label.split()
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_tags = None
            llm_responses.append(response_tags)
            tokens_list.append(token)
            actual_tags.append(all_actual_tags[len(llm_responses) - 1])

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "tokens": tokens_list,
            "actual_tags": actual_tags,
            "predicted_tags": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    # results_path = (
    #     RESULTS_DIR
    #     / "causal_detection"
    #     / f"{args.dataset}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    # )
    # results_path.parent.mkdir(parents=True, exist_ok=True)
    # df.to_csv(results_path, index=False)

    # logger.info(f"Inference completed. Results saved to {results_path}")

    success_rate = (df['predicted_tags'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
