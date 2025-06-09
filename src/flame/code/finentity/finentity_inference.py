from datetime import date
import pandas as pd
from flame.utils.dataset_utils import safe_load_dataset
from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import get_component_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
import litellm

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "finentity")

litellm.drop_params = True


def finentity_inference(args):
    today = date.today()
    logger.info(f"Starting FinEntity inference on {today}")

    logger.info("Loading dataset...")
    dataset = safe_load_dataset(
        "gtfintechlab/finentity", name="5768", trust_remote_code=True
    )

    # Extract sentences and actual labels
    sentences = [row["content"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["annotations"] for row in dataset["test"]]  # type: ignore

    llm_responses = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        finentity_prompt = get_prompt("finentity", PromptFormat.FEW_SHOT)
    else:
        finentity_prompt = get_prompt("finentity", PromptFormat.ZERO_SHOT)
    if finentity_prompt is None:
        raise RuntimeError("FinEntity prompt not found in registry")

    batch_size = args.batch_size
    total_batches = len(sentences) // batch_size + int(len(sentences) % batch_size > 0)
    logger.info(f"Processing {len(sentences)} sentences in {total_batches} batches.")

    # Create batches
    sentence_batches = chunk_list(sentences, batch_size)

    for batch_idx, sentence_batch in enumerate(sentence_batches):
        # Create prompt messages for the batch
        messages_batch = [
            [{"role": "user", "content": finentity_prompt(sentence)}]
            for sentence in sentence_batch
        ]

        try:
            # Process the batch
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            for response in batch_responses:
                try:
                    response_label = response.choices[0].message.content.strip()  # type: ignore
                    llm_responses.append(response_label)
                    complete_responses.append(response)
                except (KeyError, IndexError, AttributeError) as e:
                    logger.error(f"Error extracting response: {e}")
                    llm_responses.append("error")
                    complete_responses.append(None)

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            llm_responses.extend(["error"] * len(sentence_batch))
            complete_responses.extend([None] * len(sentence_batch))
            continue

    # Create the final DataFrame
    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Calculate success rate
    success_rate = (
        sum(1 for r in llm_responses if r != "error") / len(llm_responses)
    ) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
