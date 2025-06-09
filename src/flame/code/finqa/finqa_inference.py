import pandas as pd
from flame.utils.dataset_utils import safe_load_dataset
from tqdm import tqdm

from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import get_component_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "finqa")


def finqa_inference(args):
    dataset = safe_load_dataset("gtfintechlab/finqa", trust_remote_code=True)
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
    response_index = 0

    if args.prompt_format == "fewshot":
        finqa_prompt = get_prompt("finqa", PromptFormat.FEW_SHOT)
    else:
        finqa_prompt = get_prompt("finqa", PromptFormat.ZERO_SHOT)
    if finqa_prompt is None:
        raise RuntimeError("FinQA prompt not found in registry")

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

            for text, response in zip(text_batch, batch_responses):
                context.append(text)
                try:
                    response_label = response.choices[0].message.content  # type: ignore
                except Exception as e:
                    logger.debug(f"Error in response: {str(e)}\nResponse: {response}")
                    response_label = None
                llm_responses.append(response_label)
                complete_responses.append(response)
                actual_labels.append(all_actual_labels[response_index])
                response_index += 1

        except Exception as e:
            logger.debug(f"Batch {batch_idx + 1} failed: {str(e)}")
            for text in text_batch:
                context.append(text)
                llm_responses.append(None)
                complete_responses.append(None)
                actual_labels.append(all_actual_labels[response_index])
                response_index += 1

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
