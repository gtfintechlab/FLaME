"""FOMC inference module."""
import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import litellm
from pathlib import Path

from superflue.code.prompts import fomc_prompt
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.save_utils import save_inference_results
from superflue.config import LOG_DIR, LOG_LEVEL

# Configure litellm to be less verbose
litellm.set_verbose = False
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = setup_logger(
    name="fomc_inference", 
    log_file=LOG_DIR / "fomc_inference.log", 
    level=LOG_LEVEL
)

@dataclass
class InferenceConfig:
    """Configuration for FOMC inference."""
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int]
    repetition_penalty: float
    batch_size: int
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("Top_p must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def validate_sample(response: str) -> bool:
    """Validate model response format."""
    valid_labels = {"DOVISH", "HAWKISH", "NEUTRAL"}
    return response.strip().upper() in valid_labels

def load_fomc_dataset(dataset_org: str):
    """Load FOMC dataset with progress tracking."""
    logger.debug(f"Loading FOMC dataset from {dataset_org}...")
    dataset = load_dataset(f"{dataset_org}/fomc_communication", trust_remote_code=True)
    test_data = dataset["test"]
    logger.debug(f"Loaded {len(test_data)} test samples")
    return test_data

def process_batch_with_retry(args, messages_batch, batch_idx, total_batches):
    """Process a batch with litellm's retry mechanism."""
    try:
        # Using litellm's built-in retry mechanism
        batch_responses = litellm.batch_completion(
            model=args.inference_model,  # Use inference_model instead of model
            messages=messages_batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k else None,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_retries=3  # Using litellm's retry mechanism
        )
        logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses
            
    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise

def fomc_inference(args):
    """Run FOMC inference with improved logging and error handling."""
    # Extract provider and model info
    model_parts = args.inference_model.split('/')  # Use inference_model instead of model
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1]
    
    # Detailed startup logging
    logger.info(f"Starting FOMC inference with {model_name}")
    logger.debug(f"Provider: {provider}")
    logger.debug(f"Dataset organization: {args.dataset_org}")
    logger.debug(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load dataset
    test_data = load_fomc_dataset(args.dataset_org)
    
    # Initialize lists to store data
    sentences = []
    llm_responses = []
    actual_label = []
    complete_responses = []
    
    # Get all sentences and labels
    all_sentences = [item["sentence"] for item in test_data]
    all_labels = [item["label"] for item in test_data]
    
    # Create batches
    sentence_batches = chunk_list(all_sentences, args.batch_size)
    total_batches = len(sentence_batches)
    logger.info(f"Processing {len(all_sentences)} samples in {total_batches} batches")
    
    # Process batches with progress bar
    pbar = tqdm(sentence_batches, desc="Processing batches")
    for batch_idx, sentence_batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": fomc_prompt(sentence)}]
            for sentence in sentence_batch
        ]
        
        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
            
            # Process responses
            for response in batch_responses:
                response_text = response.choices[0].message.content # type: ignore
                llm_responses.append(response_text)
                complete_responses.append(response)
                actual_label.append(all_labels[len(llm_responses) - 1])
                sentences.append(all_sentences[len(llm_responses) - 1])

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
            for _ in range(args.batch_size):
                llm_responses.append(None)
                complete_responses.append(None)
                actual_label.append(None)
                sentences.append(None)
            time.sleep(10.0)
            continue

    # Create DataFrame with results
    df = pd.DataFrame({
        "sentences": sentences,
        "llm_responses": llm_responses,
        "actual_label": actual_label,
        "complete_responses": complete_responses,
    })

    # Log final statistics
    success_rate = (df['llm_responses'].notna().sum() / len(df)) * 100
    logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")
    
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
        "dataset_org": args.dataset_org,
        "success_rate": success_rate
    }
    
    # Use our new save utility
    save_inference_results(
        df=df,
        task="fomc",
        model=args.inference_model,
        metadata=metadata
    )
    
    return df