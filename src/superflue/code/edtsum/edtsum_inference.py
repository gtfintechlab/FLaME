import pandas as pd
from datasets import load_dataset
from superflue.code.inference_prompts import edtsum_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.code.tokens import tokens
from superflue.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry

logger = setup_logger(
    name="edtsum_inference", log_file=LOG_DIR / "edtsum_inference.log", level=LOG_LEVEL
)


def edtsum_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")

    dataset = load_dataset("gtfintechlab/EDTSum", trust_remote_code=True)

    test_data = dataset["test"] # type: ignore
    all_documents = [data["text"] for data in test_data] # type: ignore
    all_actual_labels = [data["answer"] for data in test_data] # type: ignore

    sentence_batches = chunk_list(all_documents, args.batch_size)
    total_batches = len(sentence_batches)

    llm_responses = []
    complete_responses = []

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, batch_content in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": edtsum_prompt(document)}]
            for document in batch_content
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(batch_content)):
                llm_responses.append(None)
                complete_responses.append(None)
            continue

        for document, response in zip(batch_content, batch_responses):
            try:
                response_label = response.choices[0].message.content # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            complete_responses.append(response)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

    df = pd.DataFrame(
        {
            "documents": all_documents,
            "llm_responses": llm_responses,
            "actual_labels": all_actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df['llm_responses'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
