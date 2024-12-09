from typing import List, Any, Optional, Dict
import time
import pandas as pd
from datetime import date
from datasets import load_dataset
from tqdm import tqdm
from litellm import completion
from ferrari.code.prompts import fiqa_task2_prompt
from ferrari.code.tokens import tokens
from ferrari.utils.logging_utils import setup_logger
from ferrari.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Set up logger
logger = setup_logger(
    name="fiqa_task2_inference",
    log_file=LOG_DIR / "fiqa_task2_inference.log",
    level=LOG_LEVEL,
)

def validate_entry(entry: Dict[str, Any]) -> bool:
    """Validate that an entry contains required fields.
    
    Args:
        entry: Dataset entry to validate
        
    Returns:
        True if entry is valid, False otherwise
    """
    required_fields = ["question", "answer"]
    return all(field in entry and entry[field] is not None for field in required_fields)

def fiqa_task2_inference(args) -> pd.DataFrame:
    """Run inference on FiQA Task 2 (financial question answering) dataset.
    
    Args:
        args: Configuration arguments including model parameters
        
    Returns:
        DataFrame containing inference results
        
    Raises:
        Exception: If dataset loading fails
    """
    logger.info("Starting FiQA Task 2 inference")
    
    # Load dataset
    logger.info("Loading dataset...")
    try:
        dataset = load_dataset("glennmatlin/FiQA_Task2", split="test", trust_remote_code=True)
        total_items = len(dataset)  # type: ignore
        logger.info(f"Loaded {total_items} test samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Initialize storage
    questions: List[str] = []
    llm_responses: List[Optional[str]] = []
    actual_answers: List[str] = []
    complete_responses: List[Any] = []
    skipped_entries = 0

    # Process each entry
    for i, entry in enumerate(tqdm(dataset, desc="Processing entries")):  # type: ignore
        # Validate entry
        if not validate_entry(entry):  # type: ignore
            logger.warning(f"Skipping invalid entry at index {i}")
            skipped_entries += 1
            continue

        try:
            # Extract question and actual answer
            question = entry["question"]  # type: ignore
            actual_answer = entry["answer"]  # type: ignore
            
            # Store instance data
            questions.append(question)
            actual_answers.append(actual_answer)

            # Get model response
            logger.debug(f"Processing entry {i+1}/{total_items}")
            model_response = completion(
                messages=[{"role": "user", "content": fiqa_task2_prompt(question)}],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model)
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

    if skipped_entries > 0:
        logger.warning(f"Skipped {skipped_entries} invalid entries")
    
    # Create results DataFrame
    df = pd.DataFrame(
        {
            "question": questions,
            "llm_responses": llm_responses,
            "actual_answers": actual_answers,
            "complete_responses": complete_responses,
        }
    )

    # Save results
    results_path = (
        RESULTS_DIR
        / "fiqa2"
        / f"{args.dataset}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")
    logger.info(f"Processed {len(df)} entries with {len([r for r in llm_responses if r is not None])} successful responses")

    return df 