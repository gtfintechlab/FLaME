import time
from datetime import date
import pandas as pd
from datasets import load_dataset
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list
import litellm
from typing import Dict, Any, List, Optional, Tuple
from litellm import completion 
from superflue.code.inference_prompts import refind_prompt
from superflue.code.tokens import tokens

from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from tqdm import tqdm

# Setup logger for ReFinD inference
logger = setup_logger(
    name="refind_inference",
    log_file=LOG_DIR / "refind_inference.log",
    level=LOG_LEVEL,
)


def refind_inference(args):
    
    today = date.today()
    logger.info(f"Starting ReFinD inference on {today}")

    # Load the ReFinD dataset (test split)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/ReFinD", trust_remote_code=True)

    results_path = (
        RESULTS_DIR
        / "refind"
        / f"refind_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    test_data = dataset["test"]  # type: ignore
    all_sentences = [' '.join(['[ENT1]'] + sample['token'][sample['e1_start']:sample['e1_end']] + ['[/ENT1]'] + sample['token'][sample['e1_end']+1:sample['e2_start']] + ['[ENT2]'] + sample['token'][sample['e2_start']:sample['e2_end']] + ['[/ENT2]']) for sample in test_data]  # type: ignore
    all_actual_labels = [sample["rel_group"] for sample in test_data]  # type: ignore

    sentence_batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(sentence_batches)

    # Initialize lists to store entities, actual labels, model responses, and complete responses
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on ReFinD with model {args.model}...")

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": refind_prompt(sentence)}]
            for sentence in sentence_batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in sentence_batch:
                sentences.append(None)
                complete_responses.append(None)
                llm_responses.append(None)
                actual_labels.append(None)
            continue

        # Process responses
        for sentence, response in zip(sentence_batch, batch_responses):
            sentences.append(sentence)
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
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Save the results to a CSV file
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    success_rate = (df['llm_responses'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df