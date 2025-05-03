import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from superflue.code.inference_prompts import fpb_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_DIR, LOG_LEVEL
from superflue.utils.batch_utils import process_batch_with_retry, chunk_list

logger = setup_logger(
    name="fpb_inference", log_file=LOG_DIR / "fpb_inference.log", level=LOG_LEVEL
)


def fpb_inference(args):
    task = args.dataset.strip('“”"')
    logger.info(f"Starting inference for {task} using model {args.model}.")
    dataset = load_dataset(
        "gtfintechlab/financial_phrasebank_sentences_allagree",
        "5768",
        trust_remote_code=True,
    )

    llm_responses = []
    complete_responses = []

    test_data = dataset["test"]  # type: ignore
    all_sentences = [data["sentence"] for data in test_data]  # type: ignore
    all_actual_labels = [data["label"] for data in test_data]  # type: ignore

    batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch_content in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": fpb_prompt(sentence)}]
            for sentence in batch_content
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in batch_content:
                complete_responses.append(None)
                llm_responses.append(None)
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
            "sentences": all_sentences,
            "llm_responses": llm_responses,
            "actual_labels": all_actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
