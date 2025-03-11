import pandas as pd
from datasets import load_dataset
from superflue.code.inference_prompts import ectsum_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from tqdm import tqdm

# Setup logger for ectsum inference
logger = setup_logger(
    name="ectsum_inference", log_file=LOG_DIR / "ectsum_inference.log", level=LOG_LEVEL
)

def ectsum_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")

    dataset = load_dataset("gtfintechlab/ECTSum", trust_remote_code=True)

    documents = [row["context"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["response"] for row in dataset["test"]]  # type: ignore
    llm_responses = []
    complete_responses = []
    
    total_batches = len(documents) // args.batch_size + int(len(documents) % args.batch_size > 0)

    logger.info(f"Processing {len(documents)} documents in {total_batches} batches.")

    document_batches = chunk_list(documents, args.batch_size)
    
    pbar = tqdm(document_batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch processing
        messages_batch = [
            [{"role": "user", "content": ectsum_prompt(doc)}] for doc in batch
        ]

        try:
            # Use batch processing with retry mechanism
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            for _ in batch:
                llm_responses.append("Error")
                complete_responses.append(None)
            continue

        for response in batch_responses:
            complete_responses.append(response)
            try:
                response_text = response.choices[0].message.content.strip()  # type: ignore
                llm_responses.append(response_text)
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting response: {e}")
                llm_responses.append("Error")
        
        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")
        
    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df['llm_responses'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
