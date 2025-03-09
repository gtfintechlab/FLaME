from datetime import date
import pandas as pd
from datasets import load_dataset
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list
from superflue.code.inference_prompts import refind_prompt
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
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/ReFinD", trust_remote_code=True)

    test_data = dataset["test"]  # type: ignore
    all_sentences = [' '.join(['[ENT1]'] + sample['token'][sample['e1_start']:sample['e1_end']] + ['[/ENT1]'] + sample['token'][sample['e1_end']+1:sample['e2_start']] + ['[ENT2]'] + sample['token'][sample['e2_start']:sample['e2_end']] + ['[/ENT2]']) for sample in test_data]  # type: ignore
    all_actual_labels = [sample["rel_group"] for sample in test_data]  # type: ignore

    batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(batches)

    # Initialize lists to store entities, actual labels, model responses, and complete responses
    llm_responses = []
    complete_responses = []

    logger.info(f"Starting inference on ReFinD with model {args.model}...")

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": refind_prompt(sentence)}]
            for sentence in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in batch:
                complete_responses.append(None)
                llm_responses.append(None)
            continue

        for response in batch_responses:
            complete_responses.append(response)
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
        
        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")
        
    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "sentences": all_sentences,
            "llm_responses": llm_responses,
            "actual_labels": all_actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df['llm_responses'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df