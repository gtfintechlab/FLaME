from flame.utils.batch_utils import process_batch_with_retry, chunk_list
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from flame.code.inference_prompts import fomc_prompt
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="fomc_inference", log_file=LOG_DIR / "fomc_inference.log", level=LOG_LEVEL
)


def fomc_inference(args):
    """Run FOMC inference with improved logging and error handling."""
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/fomc_communication", trust_remote_code=True)
    test_data = dataset["test"]  # type: ignore

    # Initialize result containers
    llm_responses = []
    complete_responses = []

    # Get all sentences and labels
    all_sentences = [item["sentence"] for item in test_data]  # type: ignore
    all_labels = [item["label"] for item in test_data]  # type: ignore

    # Create batches
    sentence_batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(sentence_batches)
    logger.info(f"Processing {len(all_sentences)} samples in {total_batches} batches")

    # Process batches with progress bar
    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": fomc_prompt(sentence)}]
            for sentence in sentence_batch
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
                complete_responses.append(None)
                llm_responses.append(None)
            continue

        # Process responses
        for sentence, response in zip(sentence_batch, batch_responses):
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = "Error"

            llm_responses.append(response_label)
            complete_responses.append(response)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    # Create results DataFrame
    df = pd.DataFrame(
        {
            "sentences": all_sentences,
            "llm_responses": llm_responses,
            "actual_labels": all_labels,
            "complete_responses": complete_responses,
        }
    )

    # Log final statistics
    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
