import pandas as pd
from datasets import load_dataset
from superflue.code.inference_prompts import finred_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list
from superflue.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm

# Setup logger for FinRED inference
logger = setup_logger(
    name="finred_inference", log_file=LOG_DIR / "finred_inference.log", level=LOG_LEVEL
)


def finred_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/FinRed", trust_remote_code=True)

    # Initialize lists to store sentences, actual labels, model responses, and complete responses
    llm_responses = []
    complete_responses = []

    test_data = dataset["test"]  # type: ignore
    all_inputs = [(data["sentence"], data["entities"]) for data in test_data]  # type: ignore
    all_inputs = [
        (input[0], entity_pair) for input in all_inputs for entity_pair in input[1]
    ]
    all_actual_labels = [data["relations"] for data in test_data]  # type: ignore
    all_actual_labels = [label for labels in all_actual_labels for label in labels]

    batches = chunk_list(all_inputs, args.batch_size)
    total_batches = len(batches)

    logger.info(f"Starting inference on FinRED with model {args.model}...")

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [
                {
                    "role": "user",
                    "content": finred_prompt(input[0], input[1][0], input[1][1]),
                }
            ]
            for input in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Add None values for failed batch
            for _ in batch:
                complete_responses.append(None)
                llm_responses.append(None)
            continue

        # Process responses
        for response in batch_responses:
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            complete_responses.append(response)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "sentence": [input[0] for input in all_inputs],
            "entity_pairs": [input[1] for input in all_inputs],
            "actual_labels": all_actual_labels,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
