"""Batch processing utilities for LLM inference."""

import time
import logging
import json
import os
import warnings
from typing import List, Any, Callable, Dict, Optional, Tuple
from litellm import batch_completion
import litellm  # Import litellm directly to configure it globally

from superflue.utils.logging_utils import get_logger

# Suppress warnings at this level too for redundancy
warnings.filterwarnings("ignore", message=".*together.*", category=Warning)
warnings.filterwarnings("ignore", message=".*function.*calling.*", category=Warning)
warnings.filterwarnings("ignore", message=".*response format.*", category=Warning)

# Get logger for this module
logger = get_logger(__name__)


def litellm_logger_fn(model_call_dict: Dict) -> None:
    """Custom logger function for LiteLLM to redirect its logs to our logger.

    This function aggressively filters out warnings and only logs errors by default.
    Debug logs are only shown if LITELLM_LOG=DEBUG is set.
    """
    # Skip all warning-level messages
    if model_call_dict.get("level") == "warning":
        return

    # Only log errors by default
    litellm_level = os.getenv("LITELLM_LOG", "ERROR")
    if litellm_level != "DEBUG" and model_call_dict.get("level") != "error":
        return

    # For errors and debug (when enabled), log the message
    log_level = (
        logging.ERROR if model_call_dict.get("level") == "error" else logging.DEBUG
    )

    # Format the message more concisely for cleaner output
    if isinstance(model_call_dict.get("message"), str):
        logger.log(log_level, f"LiteLLM: {model_call_dict['message']}")
    else:
        logger.log(log_level, f"LiteLLM: {json.dumps(model_call_dict, default=str)}")


# Configure LiteLLM to use our logger globally
litellm.set_verbose = False  # Disable built-in printing
litellm.logger_fn = litellm_logger_fn


def log_debug_info(prefix: str, obj: Any) -> None:
    """Helper to log detailed debug information."""
    if logger.getEffectiveLevel() > logging.DEBUG:
        return  # Skip if we're not in debug mode

    try:
        if hasattr(obj, "__dict__"):
            obj_dict = obj.__dict__
        elif isinstance(obj, (list, dict)):
            obj_dict = obj
        else:
            obj_dict = {"value": str(obj)}

        logger.debug(f"{prefix}:\n{json.dumps(obj_dict, indent=2, default=str)}")
    except Exception as e:
        logger.debug(f"Error logging debug info: {str(e)}")
        logger.debug(f"{prefix}: {str(obj)}")


def process_batch_with_retry(
    args,
    messages_batch: List[Dict],
    batch_idx: int,
    total_batches: int,
    max_retries: int = 3,
    base_wait: float = 10.0,
) -> List[Any]:
    """Process a batch of messages with retry mechanism for LLM completions."""
    logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")

    # TODO: This function does not currently have a hard guarantee it will use the right model for extraction/inference. If an extraction model is passed, it will use the inference model for the extraction. This is a major problem!!!
    extraction_model = getattr(args, "extraction_model", None)
    inference_model = getattr(args, "inference_model", None)

    if extraction_model and inference_model:
        model_name = extraction_model
    elif inference_model and not extraction_model:
        model_name = inference_model
    elif extraction_model and not inference_model:
        raise ValueError(
            "Extraction model found in args, but no inference model found."
        )
    else:
        raise ValueError(
            "Neither extraction nor inference model names were found in args"
        )

    # Only log configuration in debug mode
    if logger.getEffectiveLevel() <= logging.DEBUG:
        log_debug_info("Args", args)
        logger.debug(
            f"First message in batch: {json.dumps(messages_batch[0], indent=2)}"
        )
        logger.debug(f"Total messages in batch: {len(messages_batch)}")

    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(
                    f"Attempt {retry_count + 1}/{max_retries} for batch {batch_idx + 1}"
                )

            batch_responses = batch_completion(
                model=model_name,
                messages=messages_batch,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )

            # Only log responses in debug mode
            if logger.getEffectiveLevel() <= logging.DEBUG:
                for i, resp in enumerate(batch_responses):
                    log_debug_info(f"Response {i}", resp)

            # Validate responses
            if not batch_responses:
                raise ValueError("Empty response from LLM")

            # Check each response
            for i, resp in enumerate(batch_responses):
                if not hasattr(resp, "choices") or not resp.choices:
                    logger.warning(
                        f"Invalid response format for message {i} in batch {batch_idx + 1}"
                    )
                    if logger.getEffectiveLevel() <= logging.DEBUG:
                        log_debug_info("Invalid response structure", resp)

            logger.info(f"Successfully processed batch {batch_idx + 1}/{total_batches}")
            return batch_responses

        except Exception as e:
            retry_count += 1
            wait_time = base_wait * (2 ** (retry_count - 1))

            logger.error(
                f"Error processing batch {batch_idx + 1} (attempt {retry_count}): {type(e).__name__} - {str(e)}"
            )

            if retry_count == max_retries:
                last_error = e
                break

            logger.info(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

    error_msg = f"All {max_retries} attempts failed for batch {batch_idx + 1}"
    if last_error:
        error_msg += f": {str(last_error)}"
    raise Exception(error_msg)


def process_batch_responses(
    batch_responses: List[Any],
    validator_fn: Callable[[str], bool],
    logger: logging.Logger,
    batch_idx: int,
    labels: Optional[List[Any]] = None,
    sentences: Optional[List[str]] = None,
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """Process batch responses with validation and error handling."""
    llm_responses = []
    complete_responses = []
    actual_labels = []
    processed_sentences = []

    logger.debug(
        f"Processing {len(batch_responses)} responses from batch {batch_idx + 1}"
    )

    for i, response in enumerate(batch_responses):
        log_debug_info(f"Processing response {i} in batch {batch_idx + 1}", response)

        if hasattr(response, "choices") and response.choices:
            response_text = response.choices[0].message.content
            logger.debug(f"Response {i} text: {response_text}")

            if validator_fn(response_text):
                logger.debug(f"Response {i} passed validation")
                llm_responses.append(response_text)
                complete_responses.append(response)
                if labels:
                    actual_labels.append(labels[i])
                if sentences:
                    processed_sentences.append(sentences[i])
            else:
                logger.warning(f"Response {i} failed validation: {response_text}")
                llm_responses.append(None)
                complete_responses.append(None)
                if labels:
                    actual_labels.append(None)
                if sentences:
                    processed_sentences.append(None)
        else:
            logger.warning(
                f"Invalid response structure in batch {batch_idx + 1} for item {i}"
            )
            log_debug_info("Invalid response", response)
            llm_responses.append(None)
            complete_responses.append(None)
            if labels:
                actual_labels.append(None)
            if sentences:
                processed_sentences.append(None)

    return llm_responses, complete_responses, actual_labels, processed_sentences


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into smaller chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
