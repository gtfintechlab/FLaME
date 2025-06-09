import pandas as pd
from tqdm import tqdm

from flame.utils.dataset_utils import safe_load_dataset
from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import get_component_logger
from flame.utils.batch_utils import chunk_list, process_batch_with_retry

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "tatqa")


def tatqa_inference(args):
    dataset = safe_load_dataset("gtfintechlab/TATQA", trust_remote_code=True)

    # Initialize lists to store context, model responses, actual answers, and complete responses
    context = []
    llm_responses = []
    actual_answers = []
    complete_responses = []

    tatqa_prompt = get_prompt("tatqa", PromptFormat.ZERO_SHOT)
    if tatqa_prompt is None:
        raise RuntimeError("TATQA prompt not found in registry")

    # Prepare all data first
    all_prompts = []
    for entry in dataset["test"]:  # type: ignore
        question = entry["query"]  # type: ignore
        context_text = entry["text"]  # type: ignore
        combined_text = f"{context_text} {question}"  # Combine context and question
        context.append(combined_text)

        actual_answer = entry["answer"]  # type: ignore
        actual_answers.append(actual_answer)

        all_prompts.append(tatqa_prompt(combined_text))

    logger.info(
        f"Processing {len(all_prompts)} TATQA examples in batches of {args.batch_size}"
    )

    # Process in batches
    batches = list(chunk_list(all_prompts, args.batch_size))
    for batch_idx, batch_prompts in enumerate(
        tqdm(batches, desc="Processing TATQA batches")
    ):
        # Convert prompts to messages format for batch processing
        messages_batch = [
            [{"role": "user", "content": prompt}] for prompt in batch_prompts
        ]
        batch_responses = process_batch_with_retry(
            args, messages_batch, batch_idx, len(batches)
        )

        complete_responses.extend(batch_responses)
        # Extract text responses
        for response in batch_responses:
            if response:
                llm_responses.append(response.choices[0].message.content)  # type: ignore
            else:
                llm_responses.append(None)

    df = pd.DataFrame(
        {
            "context": context,
            "response": llm_responses,
            "actual_answer": actual_answers,
            "complete_responses": complete_responses,
        }
    )

    return df
