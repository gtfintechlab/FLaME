import pandas as pd
from datasets import load_dataset
from flame.code.prompts_zeroshot import edtsum_zeroshot_prompt
from flame.code.prompts_fewshot import edtsum_fewshot_prompt
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm
import litellm
from typing import Any, List

logger = setup_logger(
    name="edtsum_inference", log_file=LOG_DIR / "edtsum_inference.log", level=LOG_LEVEL
)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_batch_with_retry(args, messages_batch, batch_idx, total_batches):
    """Process a batch with litellm's retry mechanism."""
    try:
        # Using litellm's built-in retry mechanism
        batch_responses = litellm.batch_completion(
            model=args.model,
            messages=messages_batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            # top_k=args.top_k if args.top_k else None,
            top_p=args.top_p,
            # repetition_penalty=args.repetition_penalty,
            num_retries=3,  # Using litellm's retry mechanism
        )
        logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses

    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise


def edtsum_inference(args):
    # today = date.today()

    dataset = load_dataset("gtfintechlab/EDTSum", trust_remote_code=True)

    test_data = dataset["test"]  # type: ignore
    all_documents = [data["text"] for data in test_data]  # type: ignore
    all_actual_labels = [data["answer"] for data in test_data]  # type: ignore

    sentence_batches = chunk_list(all_documents, args.batch_size)
    total_batches = len(sentence_batches)

    documents = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    if args.prompt_format == "fewshot":
        edtsum_prompt = edtsum_fewshot_prompt
    elif args.prompt_format == "zeroshot":
        edtsum_prompt = edtsum_zeroshot_prompt

    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, batch_content in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": edtsum_prompt(document)}]
            for document in batch_content
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(len(batch_content)):
                documents.append(None)
                llm_responses.append(None)
                complete_responses.append(None)
                actual_labels.append(None)
            continue

        for document, response in zip(batch_content, batch_responses):
            documents.append(document)
            try:
                response_label = response.choices[0].message.content  # type: ignore
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                response_label = None
            llm_responses.append(response_label)
            complete_responses.append(response)
            actual_labels.append(all_actual_labels[len(llm_responses) - 1])

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")

    df = pd.DataFrame(
        {
            "documents": documents,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    success_rate = (df["llm_responses"].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")

    return df
