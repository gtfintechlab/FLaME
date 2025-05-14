from typing import Any, List
from litellm import batch_completion
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_LEVEL, LOG_DIR

logger = setup_logger(
    name="batch_utils",
    log_file=LOG_DIR / "batch_utils.log",
    level=LOG_LEVEL,
)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into smaller chunks of specified size.

    Args:
        lst: The input list to be chunked
        chunk_size: The size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_batch_with_retry(
    args, messages_batch, batch_idx, total_batches, max_tokens=None
):
    """Process a batch with litellm's retry mechanism.

    Args:
        args: Arguments containing model configuration
        messages_batch: List of message batches to process
        batch_idx: Current batch index
        total_batches: Total number of batches
        max_tokens: Optional override for max_tokens (if None, uses args.max_tokens)

    Returns:
        Batch responses from the LLM

    Raises:
        Exception: If batch processing fails after retries
    """
    try:
        # Build base parameters
        params = {
            "model": args.model,
            "messages": messages_batch,
            "max_tokens": max_tokens if max_tokens is not None else args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_retries": 3,  # Using litellm's retry mechanism
        }

        # Add optional parameters if they exist in args
        if hasattr(args, "top_k") and args.top_k:
            params["top_k"] = args.top_k

        if hasattr(args, "repetition_penalty"):
            params["repetition_penalty"] = args.repetition_penalty

        batch_responses = batch_completion(**params)
        logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses
    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise
