from datetime import date
import pandas as pd
from datasets import load_dataset
from superflue.code.prompts_zeroshot import finer_zeroshot_prompt
from superflue.code.prompts_fewshot import finer_fewshot_prompt
import litellm

# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="finer_inference", log_file=LOG_DIR / "finer_inference.log", level=LOG_LEVEL
)

litellm.drop_params = True


def finer_inference(args):
    today = date.today()
    logger.info(f"Starting FinER inference on {today}")

    # Load the dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/finer-ord-bio", trust_remote_code=True)

    # Extract data
    sentences = [row["tokens"] for row in dataset["test"]]  # type: ignore
    actual_labels = [row["tags"] for row in dataset["test"]]  # type: ignore

    llm_responses = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        finer_prompt = finer_fewshot_prompt
    elif args.prompt_format == "zeroshot":
        finer_prompt = finer_zeroshot_prompt

    batch_size = args.batch_size
    total_batches = len(sentences) // batch_size + int(len(sentences) % batch_size > 0)
    logger.info(f"Processing {len(sentences)} sentences in {total_batches} batches.")

    # Create batches
    sentence_batches = chunk_list(sentences, batch_size)

    for batch_idx, sentence_batch in enumerate(sentence_batches):
        # Create prompt messages for the batch
        messages_batch = [
            [{"role": "user", "content": finer_prompt(sentence)}]
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
    # Save results to a CSV file
    results_path = (
        RESULTS_DIR / "finer" / f"finer_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
