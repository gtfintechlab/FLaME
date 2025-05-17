from datetime import date
import pandas as pd
from datasets import load_dataset
from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_LEVEL, LOG_DIR, RESULTS_DIR
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
import litellm

logger = setup_logger(
    name="causal_classification_inference",
    log_file=LOG_DIR / "causal_classification_inference.log",
    level=LOG_LEVEL,
)

litellm.drop_params = True


def causal_classification_inference(args):
    today = date.today()
    logger.info(f"Starting Causal Classification inference on {today}")

    # Load the dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/CausalClassification", trust_remote_code=True)

    # Extract data from the test split
    texts = [row["text"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["label"] for row in dataset["test"]]  # type: ignore
    llm_responses = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        causal_classification_prompt = get_prompt(
            "causal_classification", PromptFormat.FEW_SHOT
        )
    else:
        causal_classification_prompt = get_prompt(
            "causal_classification", PromptFormat.ZERO_SHOT
        )
    if causal_classification_prompt is None:
        raise RuntimeError("Causal Classification prompt not found in registry")

    batch_size = args.batch_size
    total_batches = len(texts) // batch_size + int(len(texts) % batch_size > 0)
    logger.info(f"Processing {len(texts)} texts in {total_batches} batches.")

    # Create batches
    text_batches = chunk_list(texts, batch_size)

    for batch_idx, text_batch in enumerate(text_batches):
        # Create prompt messages for the batch
        messages_batch = [
            [{"role": "user", "content": causal_classification_prompt(text)}]
            for text in text_batch
        ]

        try:
            # Process the batch
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
            # time.sleep(1)

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
            llm_responses.extend(["error"] * len(text_batch))
            complete_responses.extend([None] * len(text_batch))
            continue

    # Create the final DataFrame
    df = pd.DataFrame(
        {
            "texts": texts,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Save results to a CSV file
    results_path = (
        RESULTS_DIR
        / "causal_classification"
        / f"causal_classification_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
