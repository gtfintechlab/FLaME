import pandas as pd
from datasets import load_dataset
from superflue.code.inference_prompts import causal_detection_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from tqdm import tqdm

logger = setup_logger(
    name="cd_inference", log_file=LOG_DIR / "cd_inference.log", level=LOG_LEVEL
)

def casual_detection_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")

    dataset = load_dataset("gtfintechlab/CausalDetection", trust_remote_code=True)

    test_data = dataset["test"]  # type: ignore
    all_tokens = [data["tokens"] for data in test_data]  # type: ignore
    all_actual_tags = [data["tags"] for data in test_data]  # type: ignore

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
                complete_responses.append(None)
                llm_responses.append(None)
            continue
        
        for token, response in zip(token_batch, batch_responses):
            complete_responses.append(response)
            try: 
                response_tags = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_tags = None
            llm_responses.append(response_tags)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df = pd.DataFrame(
        {
            "tokens": all_tokens,
            "actual_tags": all_actual_tags,
            "predicted_tags": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df['predicted_tags'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
