from datetime import date
import pandas as pd
from flame.utils.dataset_utils import safe_load_dataset
from tqdm import tqdm

from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import get_component_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "refind")


def refind_inference(args):
    today = date.today()
    logger.info(f"Starting ReFinD inference on {today}")

    # Load the ReFinD dataset (test split)
    logger.info("Loading dataset...")
    dataset = safe_load_dataset("gtfintechlab/ReFinD", trust_remote_code=True)

    # We'll generate the filename at the end of the function

    test_data = dataset["test"]  # type: ignore
    all_sentences = [
        " ".join(
            ["[ENT1]"]
            + sample["token"][sample["e1_start"] : sample["e1_end"]]
            + ["[/ENT1]"]
            + sample["token"][sample["e1_end"] + 1 : sample["e2_start"]]
            + ["[ENT2]"]
            + sample["token"][sample["e2_start"] : sample["e2_end"]]
            + ["[/ENT2]"]
        )
        for sample in test_data
    ]  # type: ignore
    all_actual_labels = [sample["rel_group"] for sample in test_data]  # type: ignore

    sentence_batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(sentence_batches)

    # Initialize lists to store entities, actual labels, model responses, and complete responses
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        refind_prompt = get_prompt("refind", PromptFormat.FEW_SHOT)
    else:
        refind_prompt = get_prompt("refind", PromptFormat.ZERO_SHOT)
    if refind_prompt is None:
        raise RuntimeError("ReFinD prompt not found in registry")

    logger.info(f"Starting inference on ReFinD with model {args.model}...")

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": refind_prompt(sentence)}]
            for sentence in sentence_batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in sentence_batch:
                sentences.append(None)
                complete_responses.append(None)
                llm_responses.append(None)
                actual_labels.append(None)
            continue

        # Process responses
        for sentence, response in zip(sentence_batch, batch_responses):
            sentences.append(sentence)
            complete_responses.append(response)
            try:
                response_label = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            actual_labels.append(all_actual_labels[len(llm_responses) - 1])
        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    # Create the final DataFrame after the loop
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
