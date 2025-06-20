from datetime import date

import pandas as pd
from tqdm import tqdm

from flame.code.prompts import PromptFormat, get_prompt
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.dataset_utils import safe_load_dataset
from flame.utils.logging_utils import get_component_logger

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "fiqa_task1")


def fiqa_task1_inference(args):
    today = date.today()
    logger.info(f"Starting FiQA Task 1 inference on {today}")

    # Load dataset and initialize storage for results
    logger.info("Loading dataset...")
    dataset = safe_load_dataset("gtfintechlab/FiQA_Task1", trust_remote_code=True)

    test_data = dataset["test"]  # type: ignore
    all_texts = [
        f"Sentence: {data['sentence']}. Snippets: {data['snippets']}. Target aspect: {data['target']}"
        for data in test_data
    ]  # type: ignore
    all_targets = [data["target"] for data in test_data]  # type: ignore
    all_sentiments = [data["sentiment_score"] for data in test_data]  # type: ignore

    sentence_batches = chunk_list(all_texts, args.batch_size)
    total_batches = len(sentence_batches)
    context = []
    llm_responses = []
    actual_targets = []
    actual_sentiments = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        fiqa_task1_prompt = get_prompt("fiqa_task1", PromptFormat.FEW_SHOT)
    else:
        fiqa_task1_prompt = get_prompt("fiqa_task1", PromptFormat.ZERO_SHOT)
    if fiqa_task1_prompt is None:
        raise RuntimeError("FiQA Task1 prompt not found in registry")

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": fiqa_task1_prompt(sentence)}]
            for sentence in sentence_batch
        ]
        try:
            # Process batch with retry mechanism
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(sentence_batch)):
                llm_responses.append(None)
                complete_responses.append(None)
                context.append(None)
                actual_targets.append(None)
                actual_sentiments.append(None)

        for sentence, response in zip(sentence_batch, batch_responses):
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            complete_responses.append(response)
            context.append(sentence)
            actual_targets.append(all_targets[len(llm_responses) - 1])
            actual_sentiments.append(all_sentiments[len(llm_responses) - 1])
        pbar.set_description(f"Completed batch {batch_idx + 1}/{total_batches}")

    # Create DataFrame with results
    df = pd.DataFrame(
        {
            "context": context,
            "llm_responses": llm_responses,
            "actual_target": actual_targets,
            "actual_sentiment": actual_sentiments,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
