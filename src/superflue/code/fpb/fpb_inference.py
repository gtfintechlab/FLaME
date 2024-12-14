import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from superflue.code.prompts import fpb_prompt
from superflue.utils.logging_utils import setup_logger
from superflue.utils.save_utils import save_inference_results
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.config import LOG_DIR, LOG_LEVEL

# Configure logging
logger = setup_logger(
    name="fpb_inference", log_file=LOG_DIR / "fpb_inference.log", level=LOG_LEVEL
)

data_seed = "5768"


def fpb_inference(args):
    """Run inference for the Financial PhraseBank task.

    Args:
        args: Command line arguments containing model configuration

    Returns:
        DataFrame containing inference results
    """
    # Extract provider and model info
    model_parts = args.inference_model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1]

    # Detailed startup logging
    logger.info(f"Starting FPB inference with model: {args.inference_model}")
    logger.debug("Model Configuration:")
    logger.debug(f"- Provider: {provider}")
    logger.debug(f"- Model Name: {model_name}")
    logger.debug(f"- Temperature: {args.temperature}")
    logger.debug(f"- Top P: {args.top_p}")
    logger.debug(f"- Top K: {args.top_k}")
    logger.debug(f"- Max Tokens: {args.max_tokens}")
    logger.debug(f"- Batch Size: {args.batch_size}")
    logger.debug(f"- Repetition Penalty: {args.repetition_penalty}")
    logger.debug(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info("Loading dataset...")
    dataset = load_dataset(
        "gtfintechlab/financial_phrasebank_sentences_allagree",
        data_seed,
        trust_remote_code=True,
    )
    test_data = dataset["test"]
    logger.debug(f"Loaded {len(test_data)} test samples")

    # Initialize lists to store data
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    # Get all sentences and labels
    all_sentences = [item["sentence"] for item in test_data]
    all_labels = [item["label"] for item in test_data]
    logger.debug(f"Total samples to process: {len(all_sentences)}")

    # Create batches
    sentence_batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(sentence_batches)
    logger.info(f"Processing {len(all_sentences)} samples in {total_batches} batches")

    # Process batches with progress bar
    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}")
        # Prepare messages for batch
        messages_batch = [
            [
                {
                    "role": "user",
                    "content": fpb_prompt(
                        sentence,
                        prompt_format=getattr(args, "prompt_format", "superflue"),
                    ),
                }
            ]
            for sentence in sentence_batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            # Process responses
            for response in batch_responses:
                if hasattr(response, "choices") and response.choices:
                    response_text = response.choices[0].message.content
                    llm_responses.append(response_text)
                    complete_responses.append(response)
                    actual_labels.append(all_labels[len(llm_responses) - 1])
                    sentences.append(all_sentences[len(llm_responses) - 1])
                else:
                    logger.warning(f"Invalid response format in batch {batch_idx + 1}")
                    llm_responses.append(None)
                    complete_responses.append(None)
                    actual_labels.append(None)
                    sentences.append(None)

            pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed with error: {str(e)}")
            # Add None values for failed batch
            for _ in range(len(sentence_batch)):
                llm_responses.append(None)
                complete_responses.append(None)
                actual_labels.append(None)
                sentences.append(None)
            logger.debug(f"Added {len(sentence_batch)} None values for failed batch")
            time.sleep(10.0)  # Wait before continuing to next batch
            continue

    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Calculate success rate
    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")
    logger.debug(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results with metadata
    metadata = {
        "model": args.inference_model,
        "provider": provider,
        "model_name": model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "repetition_penalty": args.repetition_penalty,
        "success_rate": success_rate,
        "dataset": "financial_phrasebank_sentences_allagree",
    }

    save_inference_results(
        df=df, task="fpb", model=args.inference_model, metadata=metadata
    )

    return df
