from typing import List, Any, Optional
import time
import pandas as pd
from datetime import date
from datasets import load_dataset
from tqdm import tqdm
from litellm import completion
from ferrari.code.prompts import fiqa_task1_prompt
from ferrari.code.tokens import tokens
from ferrari.utils.logging_utils import setup_logger
from ferrari.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Set up logger
logger = setup_logger(
    name="fiqa_task1_inference",
    log_file=LOG_DIR / "fiqa_task1_inference.log",
    level=LOG_LEVEL,
)

def fiqa_task1_inference(args) -> pd.DataFrame:
    """Run inference on FiQA Task 1 (aspect-based sentiment analysis) dataset.
    
    Args:
        args: Configuration arguments including model parameters
        
    Returns:
        DataFrame containing inference results
        
    Raises:
        Exception: If dataset loading fails
    """
    logger.info("Starting FiQA Task 1 inference")
    
    # Load dataset
    logger.info("Loading dataset...")
    try:
        dataset = load_dataset("glennmatlin/FiQA_Task1", split="test", trust_remote_code=True)
        total_items = len(dataset)  # type: ignore
        logger.info(f"Loaded {total_items} test samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Initialize storage
    context: List[str] = []
    llm_responses: List[Optional[str]] = []
    actual_targets: List[str] = []
    actual_sentiments: List[float] = []
    complete_responses: List[Any] = []

    # Process each entry
    for i, entry in enumerate(tqdm(dataset, desc="Processing entries")):  # type: ignore
        try:
            # Extract relevant fields
            sentence = entry["sentence"]  # type: ignore
            snippets = entry["snippets"]  # type: ignore
            target = entry["target"]  # type: ignore
            sentiment_score = entry["sentiment_score"]  # type: ignore

            # Store instance data
            combined_text = f"Sentence: {sentence}. Snippets: {snippets}. Target aspect: {target}"
            context.append(combined_text)
            actual_targets.append(target)
            actual_sentiments.append(sentiment_score)

            # Get model response
            logger.debug(f"Processing entry {i+1}/{total_items}")
            model_response = completion(
                messages=[{"role": "user", "content": fiqa_task1_prompt(combined_text)}],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            # Process response
            logger.debug(f"Model response: {model_response}")
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content  # type: ignore
            llm_responses.append(response_label)

        except Exception as e:
            logger.error(f"Error processing entry {i+1}: {e}")
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(10.0)
            continue
    
    # Create results DataFrame
    df = pd.DataFrame(
        {
            "context": context,
            "llm_responses": llm_responses,
            "actual_target": actual_targets,
            "actual_sentiment": actual_sentiments,
            "complete_responses": complete_responses,
        }
    )

    # Save results
    results_path = (
        RESULTS_DIR
        / "fiqa1"
        / f"{args.dataset}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df 