from litellm import batch_completion
# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_LEVEL, LOG_DIR

logger = setup_logger(
    name="batch_utils",
    log_file=LOG_DIR / "batch_utils.log",
    level=LOG_LEVEL,
)

def chunk_list(lst, chunk_size):
    """Split a list into smaller chunks of specified size.
    
    Args:
        lst: The input list to be chunked
        chunk_size: The size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def process_batch_with_retry(args, messages_batch, batch_idx, total_batches):
    """Process a batch of messages with retry mechanism for LLM completions.
    
    Args:
        args: Arguments containing model configuration
        messages_batch: List of message batches to process
        batch_idx: Current batch index
        total_batches: Total number of batches
        
    Returns:
        Batch responses from the LLM
        
    Raises:
        Exception: If batch processing fails after retries
    """
    logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
    try:
        
        batch_responses = batch_completion(
            model=args.model,
            messages=messages_batch,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            # top_k=args.top_k,
            # top_p=args.top_p,
            # repetition_penalty=args.repetition_penalty,
            num_retries=3,
            # stop=tokens(args.model),
        )
        logger.info(f"Completed batch {batch_idx + 1}/{total_batches}")
        print(batch_responses)
        return batch_responses
    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise 