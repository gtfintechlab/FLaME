import pandas as pd
from tqdm import tqdm
from flame.utils.dataset_utils import safe_load_dataset
from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import get_component_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "fpb")

# data_seed = '5768'


def fpb_inference(args):
    # TODO: (Glenn) Very low priority, we can set the data_split as configurable in yaml
    # data_splits = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
    logger.info("Starting FPB inference")
    logger.info("Loading dataset...")
    # Specify a specific data split - using '5768' as default
    dataset = safe_load_dataset(
        "gtfintechlab/financial_phrasebank_sentences_allagree",
        name="5768",  # Pass config name using 'name' parameter
        trust_remote_code=True,
    )

    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        fpb_prompt = get_prompt("fpb", PromptFormat.FEW_SHOT)
    else:
        fpb_prompt = get_prompt("fpb", PromptFormat.ZERO_SHOT)
    if fpb_prompt is None:
        raise RuntimeError("FPB prompt not found in registry")

    test_data = dataset["test"]  # type: ignore
    all_sentences = [data["sentence"] for data in test_data]  # type: ignore
    all_actual_labels = [data["label"] for data in test_data]  # type: ignore

    batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(batches)

    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch_content in enumerate(pbar):
        messages_batch = [
            [
                {
                    "role": "user",
                    "content": fpb_prompt(sentence, prompt_format="flame"),
                }
            ]
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
                actual_labels.append(None)
                sentences.append(None)

        for sentence, response in zip(batch_content, batch_responses):
            sentences.append(sentence)
            try:
                response_label = response.choices[0].message.content  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            actual_labels.append(all_actual_labels[len(llm_responses) - 1])
            complete_responses.append(response)

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
