import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from superflue.code.inference_prompts import finbench_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.config import LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="finbench_inference",
    log_file=LOG_DIR / "finbench_inference.log",
    level=LOG_LEVEL,
)


def finbench_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/finbench", trust_remote_code=True)

    # Initialize lists to store actual labels and model responses
    llm_responses = []
    complete_responses = []

    test_data = dataset["test"]  # type: ignore
    all_profiles = [data["X_profile"] for data in test_data]  # type: ignore
    all_actual_labels = [data["y"] for data in test_data]  # type: ignore

    sentence_batches = chunk_list(all_profiles, args.batch_size)
    total_batches = len(sentence_batches)

    logger.info("Starting inference on dataset...")

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": finbench_prompt(profile)}]
            for profile in sentence_batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")

            for _ in sentence_batch:
                complete_responses.append(None)
                llm_responses.append(None)
            continue

        # Process responses
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

    df = pd.DataFrame(
        {
            "X_profile": all_profiles,
            "y": all_actual_labels,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
