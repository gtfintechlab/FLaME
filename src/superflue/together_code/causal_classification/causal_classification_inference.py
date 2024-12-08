from litellm import batch_completion 
import litellm
litellm.set_verbose=True
import pandas as pd
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from datasets import load_dataset
from datetime import date
from superflue.together_code.prompts import causal_classification_prompt
from superflue.together_code.tokens import tokens
from superflue.config import LOG_LEVEL, LOG_DIR, RESULTS_DIR
from superflue.utils.logging_utils import setup_logger

logger = setup_logger(
    name="causal_classification_inference",
    log_file=LOG_DIR / "causal_classification_inference.log",
    level=LOG_LEVEL,
)
def chunk_list(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def process_batch_with_retry(args, messages_batch, batch_idx, total_batches):
    """Process a batch with retry mechanism."""
    logger.info(f"messages_batch: {messages_batch}")
    try:
        batch_responses = batch_completion(
            model=args.model,
            messages=messages_batch,
            temperature=args.temperature,
            tokens=args.max_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_retries = 3,
            stop=tokens(args.model),
        )
        logger.info(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses
    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise


def causal_classification_inference(args):
    today = date.today()
    logger.info(f"Starting Causal Classification inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/CausalClassification")

    # Initialize lists to store actual labels and model responses
    texts = [item["text"] for item in dataset["test"]]  # type: ignore
    actual_labels = [item["label"] for item in dataset["test"]]  # type: ignore
    llm_responses = []
    complete_responses = []
    
    batch_size = 10
    sentence_batches = chunk_list(texts, batch_size)
    label_batches = chunk_list(actual_labels, batch_size)
    total_batches = len(sentence_batches)

    logger.info(f"Processing {len(texts)} samples in {total_batches} batches")
    # start_t = time.time()
    for batch_idx, (sentence_batch, label_batch) in enumerate(zip(sentence_batches, label_batches)):
        messages_batch = [
            [{"role": "user", "content": causal_classification_prompt(sentence)}]
            for sentence in sentence_batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            # Process responses
            for sentence, response, actual_label in zip(sentence_batch, batch_responses, label_batch):
                complete_responses.append(response)
                logger.info(f"Respnse: {response}")
                response_label = response.choices[0].message.content
                logger.info(f"Model response: {response_label}")
                llm_responses.append(response_label)

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Append None values for the failed batch
            for _ in sentence_batch:
                complete_responses.append(None)
                llm_responses.append(None)
            continue

    df = pd.DataFrame(
        {
            "texts": texts,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )
    
    results_path = (
        RESULTS_DIR
        / "causal_classification"
        / f"causal_classification_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

    return df
