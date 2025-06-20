"""
NOTE: This task is not included in the current release.
MMLU was not used in the camera-ready version of the paper
and will be implemented in a future release.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from flame.code.mmlu.mmlu_loader import MMLULoader
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.logging_utils import get_component_logger

logger = get_component_logger("inference", "mmlu")


def format_mmlu_prompt(
    question: str,
    choices: List[str],
    examples: List[Dict],
) -> str:
    """Format the question with few-shot examples.

    Args:
        question: The question to answer
        choices: List of possible answers
        examples: List of few-shot examples

    Returns:
        Formatted prompt string
    """
    # Format examples
    examples_text = ""
    for ex in examples:
        examples_text += f"Question: {ex['question']}\n\n"
        examples_text += "Choices:\n"
        examples_text += "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(ex["choices"])
        )
        examples_text += f"\n\nAnswer: {ex['answer']}\n\n"

    # Format current question
    current_text = f"Question: {question}\n\nChoices:\n"
    current_text += "\n".join(
        f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
    )
    current_text += "\n\nAnswer:"

    return f"{examples_text}{current_text}"


def save_inference_results(df: pd.DataFrame, path: Path, metadata: Dict) -> None:
    """Save results with metadata about the run."""
    metadata_path = path.with_suffix(".meta.json")
    metadata.update(
        {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(df),
            "successful_samples": len(df[df["raw_response"].notna()]),
            "failed_samples": len(df[df["raw_response"].isna()]),
        }
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    df.to_csv(path, index=False)
    logger.info(f"Results and metadata saved to {path.parent}")


def mmlu_inference(args) -> pd.DataFrame:
    """Run MMLU inference with batching and retries.

    This function:
    1. Loads the MMLU dataset for specified subjects
    2. Processes questions in batches for efficiency
    3. Uses few-shot prompting with examples
    4. Handles errors and retries failed requests
    5. Saves results with detailed metadata

    Args:
        args: Arguments containing:
            - model: Model identifier (e.g., "together_ai/meta-llama/Llama-2-7b")
            - mmlu_subjects: List of MMLU subjects to evaluate
            - mmlu_split: Dataset split (dev/validation/test)
            - mmlu_num_few_shot: Number of few-shot examples
            - batch_size: Number of questions to process at once
            - max_tokens: Maximum tokens for model response
            - temperature: Sampling temperature
            - top_p: Nucleus sampling parameter
            - top_k: Top-k sampling parameter
            - repetition_penalty: Penalty for repeated tokens

    Returns:
        DataFrame containing:
            - question: The original question text
            - raw_response: Complete model response
            - actual_answer: Correct answer (A/B/C/D)
            - subject: Question subject area

    Raises:
        ValueError: If required arguments are missing
        Exception: If batch processing fails repeatedly

    Example:
        >>> args = argparse.Namespace(
        ...     model="together_ai/meta-llama/Llama-2-7b",
        ...     mmlu_subjects=["high_school_microeconomics"],
        ...     mmlu_split="test",
        ...     batch_size=10
        ... )
        >>> results_df = mmlu_inference(args)
    """
    # Extract provider and model info
    model_parts = args.model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1]

    # Log startup information
    logger.info(
        f"Starting MMLU inference on model '{model_name}' from provider '{provider}'"
    )
    logger.info(f"Subjects: {args.mmlu_subjects or 'default economics subjects'}")
    logger.info(f"Split: {args.mmlu_split}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load MMLU dataset
    loader = MMLULoader(
        subjects=args.mmlu_subjects,
        split=args.mmlu_split,
        num_few_shot=args.mmlu_num_few_shot or 5,
    )
    df, few_shot_examples = loader.load()
    logger.info(
        f"Loaded {len(df)} questions and {len(few_shot_examples)} few-shot examples"
    )

    # Initialize result containers
    questions = []
    raw_responses = []
    actual_answers = []
    subjects = []

    # Create batches of questions
    question_data = [
        (row["question"], row["choices"], row["answer"], row["subject"])
        for _, row in df.iterrows()
    ]
    question_batches = chunk_list(question_data, args.batch_size)
    total_batches = len(question_batches)

    # Process batches with progress bar
    pbar = tqdm(question_batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        # Prepare messages for batch
        messages_batch = [
            [{"role": "user", "content": format_mmlu_prompt(q, c, few_shot_examples)}]
            for q, c, _, _ in batch
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            pbar.set_description(
                f"Processing batches (Batch {batch_idx + 1}/{total_batches} failed)"
            )
            logger.error(f"Batch {batch_idx + 1} failed completely: {str(e)}")
            # Add None values for failed batch
            for _ in batch:
                questions.append(None)
                raw_responses.append(None)
                actual_answers.append(None)
                subjects.append(None)
            continue

        # Process responses
        for (question, choices, answer, subject), response in zip(
            batch, batch_responses
        ):
            questions.append(question)
            try:
                raw_responses.append(response.choices[0].message.content)
            except Exception as e:
                logger.error(f"Error in response: {str(e)}\nResponse: {response}")
                raw_responses.append(None)
            actual_answers.append(answer)
            subjects.append(subject)

        pbar.set_description(
            f"Processing batches (Batch {batch_idx + 1}/{total_batches} succeeded)"
        )

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "question": questions,
            "raw_response": raw_responses,
            "actual_answer": actual_answers,
            "subject": subjects,
        }
    )
    return results_df
