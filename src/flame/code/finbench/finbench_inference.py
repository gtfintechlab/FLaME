from datetime import date

import pandas as pd
from flame.utils.dataset_utils import safe_load_dataset
from tqdm import tqdm

from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="finbench_inference",
    log_file=LOG_DIR / "finbench_inference.log",
    level=LOG_LEVEL,
)


def finbench_inference(args):
    today = date.today()
    logger.info(f"Starting FinBench inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = safe_load_dataset("gtfintechlab/finbench", trust_remote_code=True)

    # Initialize lists to store actual labels and model responses
    X_profile_data = []
    y_data = []
    llm_responses = []
    complete_responses = []

    test_data = dataset["test"]  # type: ignore
    all_profiles = [data["X_profile"] for data in test_data]  # type: ignore
    all_actual_labels = [data["y"] for data in test_data]  # type: ignore

    sentence_batches = chunk_list(all_profiles, args.batch_size)
    total_batches = len(sentence_batches)

    logger.info("Starting inference on dataset...")
    # start_t = time.time()

    if args.prompt_format == "fewshot":
        finbench_prompt = get_prompt("finbench", PromptFormat.FEW_SHOT)
    else:
        finbench_prompt = get_prompt("finbench", PromptFormat.ZERO_SHOT)
    if finbench_prompt is None:
        raise RuntimeError("FinBench prompt not found in registry")

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": finbench_prompt(profile)}]
            for profile in sentence_batch
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
                X_profile_data.append(None)
                complete_responses.append(None)
                llm_responses.append(None)
                y_data.append(None)
            continue

        # Process responses
        for profile, response in zip(sentence_batch, batch_responses):
            X_profile_data.append(profile)
            complete_responses.append(response)
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            y_data.append(all_actual_labels[len(llm_responses) - 1])

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "X_profile": all_profiles,
            "y": y_data,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
