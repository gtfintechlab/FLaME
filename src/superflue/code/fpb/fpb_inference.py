import pandas as pd
import time
from tqdm import tqdm
from datasets import load_dataset
from datetime import date
from superflue.code.prompts_zeroshot import fpb_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from litellm import completion 
import litellm
from typing import Dict, Any, List, Optional, Tuple

logger = setup_logger(
    name="fpb_inference", log_file=LOG_DIR / "fpb_inference.log", level=LOG_LEVEL
)

# data_seed = '5768'

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

def fpb_inference(args):
    # TODO: (Glenn) Very low priority, we can set the data_split as configurable in yaml
    # data_splits = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
    logger.info("Starting FPB inference")
    logger.info("Loading dataset...")
    # for data_split in data_splits:
    dataset = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", data_seed, trust_remote_code=True)

    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    test_data = dataset['test'] # type: ignore
    all_sentences = [data["sentence"] for data in test_data] # type: ignore
    all_actual_labels = [data["label"] for data in test_data] # type: ignore

    batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch_content in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": fpb_prompt(sentence, prompt_format='superflue')}]
            for sentence in batch_content]
        try:
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, total_batches)
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in batch_content:
                complete_responses.append(None)
                llm_responses.append(None)
                actual_labels.append(None)
                sentences.append(None)
        
        for (sentence, response) in zip(batch_content, batch_responses):
            sentences.append(sentence)
            try:
                response_label = response.choices[0].message.content # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            actual_labels.append(all_actual_labels[len(llm_responses) - 1])
            complete_responses.append(response)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df['llm_responses'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

#     return df
