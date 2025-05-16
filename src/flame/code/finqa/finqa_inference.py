import pandas as pd
from datasets import load_dataset
from flame.code.prompts import finqa_zeroshot_prompt, finqa_fewshot_prompt
from flame.utils.logging_utils import setup_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm

logger = setup_logger(
    name="finqa_inference", log_file=LOG_DIR / "finqa_inference.log", level=LOG_LEVEL
)


def finqa_inference(args):
    dataset = load_dataset("gtfintechlab/finqa", trust_remote_code=True)
    test_data = dataset["test"]  # type: ignore
    all_texts = [
        f"{' '.join(data['pre_text'])} {' '.join(data['post_text'])} {' '.join([' '.join(row) for row in data['table_ori']])} {data['question']}"
        for data in test_data
    ]  # type: ignore
    all_actual_labels = [data["answer"] for data in test_data]  # type: ignore
    text_batches = chunk_list(all_texts, args.batch_size)
    total_batches = len(text_batches)

    context = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        finqa_prompt = finqa_fewshot_prompt
    elif args.prompt_format == "zeroshot":
        finqa_prompt = finqa_zeroshot_prompt

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
                context.append(None)
                llm_responses.append(None)
                complete_responses.append(None)
                actual_labels.append(None)

        for text, response in zip(text_batch, batch_responses):
            context.append(text)
            try:
                response_label = response.choices[0].message.content  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            complete_responses.append(response)
            actual_labels.append(all_actual_labels[len(llm_responses) - 1])

        pbar.set_description(f"Completed batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "context": context,
            "response": llm_responses,
            "actual_label": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["response"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
