import pandas as pd
from datasets import load_dataset
from flame.code.inference_prompts import finqa_prompt
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.batch_utils import process_batch_with_retry, chunk_list
from tqdm import tqdm

logger = setup_logger(
    name="finqa_inference", log_file=LOG_DIR / "finqa_inference.log", level=LOG_LEVEL
)


def finqa_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset("gtfintechlab/finqa", trust_remote_code=True)
    test_data = dataset["test"]  # type: ignore
    all_texts = [
        f"{' '.join(data['pre_text'])} {' '.join(data['post_text'])} {' '.join([' '.join(row) for row in data['table_ori']])} {data['question']}"
        for data in test_data
    ]  # type: ignore
    all_actual_labels = [data["answer"] for data in test_data]  # type: ignore
    text_batches = chunk_list(all_texts, args.batch_size)
    total_batches = len(text_batches)

    llm_responses = []
    complete_responses = []

    pbar = tqdm(text_batches, desc="Processing batches")
    for batch_idx, text_batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": finqa_prompt(sentence)}]
            for sentence in text_batch
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in text_batch:
                llm_responses.append(None)
                complete_responses.append(None)
            continue

        for response in batch_responses:
            try:
                response_label = response.choices[0].message.content  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            complete_responses.append(response)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df = pd.DataFrame(
        {
            "context": all_texts,
            "response": llm_responses,
            "actual_label": all_actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["response"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
