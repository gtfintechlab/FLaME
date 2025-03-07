import os
import uuid
from datetime import date
from typing import Any, Dict, List

# os.environ["LITELLM_LOG"] = "DEBUG"
import litellm

import pandas as pd
from datasets import load_dataset
from superflue.config import LOG_DIR, LOG_LEVEL, RESULTS_DIR
from superflue.code.prompts_fromferrari import economics_testbank_prompt
from superflue.utils.logging_utils import setup_logger
from tqdm import tqdm

# Set up logger
logger = setup_logger(
    name="economics_testbank",
    log_file=LOG_DIR / "economics_testbank.log",
    level=LOG_LEVEL,
)


def process_batch_with_retry(args, messages_batch, batch_idx, total_batches):
    """Process a batch with litellm's retry mechanism."""
    try:
        # Using litellm's built-in retry mechanism
        # print(messages_batch)
        batch_responses = litellm.batch_completion(
            model=args.model,
            messages=messages_batch,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k else None,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_retries=3,
        )
        logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")
        return batch_responses

    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {str(e)}")
        raise


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_refined_prompt(rationale, prompt, answer):
    """
    Refine a rationale for a given prompt using the Together API.
    """
    refined_prompt = f"Your previous response to the prompt is wrong, here is the correct answer: {answer}. Your previous reasoning was: {rationale}. Give me an updated step by step rationale in the same format as the original one. Prompt: {prompt}."
    return refined_prompt


def economics_testbank_star_inference(args):
    """
    Perform STaR inference by first running inference and then refining incorrect outputs.
    """
    today = date.today()
    logger.info(f"Starting STaR inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset(
        "acousticsmh/Combined_Economics_TestBank", trust_remote_code=True
    )["train"]
    # dataset = load_dataset("acousticsmh/Economics_TestBank", trust_remote_code=True)[
    #     "train"
    # ]

    # Initialize Together API client
    logger.info("Preparing prompts...")
    all_prompts = [economics_testbank_prompt(str(row["prompt"])) for row in dataset]
    correct_answers = [str(row["answer"]) for row in dataset]
    batch_size = 100
    prompt_batches = chunk_list(all_prompts, batch_size)
    total_batches = len(prompt_batches)

    # Step 1: Generate initial rationales
    results = []
    pbar = tqdm(prompt_batches, desc="Processing batches")
    for batch_idx, prompt_batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": prompt}] for prompt in prompt_batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
            for i, response in enumerate(batch_responses):
                prompt = prompt_batch[i]
                correct_answer = correct_answers[batch_idx * batch_size + i]
                llm_response = response.choices[0].message.content
                answer = llm_response.splitlines()[0]
                rationale = "\n".join(llm_response.splitlines()[1:])

                results.append(
                    {
                        "Prompt": prompt,
                        "Answer": answer,
                        "Correct_Answer": correct_answer,
                        "Generated_Rationale": rationale,
                    }
                )
            pbar.set_description(
                f"Processing batches (Batch {batch_idx + 1}/{total_batches} succeeded)"
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed completely: {str(e)}")
            # for prompt in prompt_batch:
            #     results.append(
            #         {
            #             "Prompt": prompt,
            #             "Answer": "No Answer",
            #             "Correct_Answer": "No Answer",
            #             "Generated_Rationale": [],
            #         }
            #     )
            continue

    logger.info("Refining incorrect rationales...")
    final_results = []
    updated_results = []
    all_prompts = []
    pbar = tqdm(prompt_batches, desc="Processing batches")
    for result in tqdm(results, desc="Refining Incorrect Responses"):
        if result["Answer"][0] != result["Correct_Answer"][0]:
            refined_prompt = get_refined_prompt(
                result["Generated_Rationale"],
                result["Prompt"],
                result["Correct_Answer"],
            )
            all_prompts.append(refined_prompt)
            updated_results.append(result)
        else:
            # print("Answer was Correct!")
            result["Refined_Rationale"] = result["Generated_Rationale"]
            final_results.append(result)

    prompt_batches = chunk_list(all_prompts, batch_size)
    total_batches = len(prompt_batches)
    pbar = tqdm(prompt_batches, desc="Processing batches")
    for batch_idx, prompt_batch in enumerate(pbar):
        messages_batch = [
            [{"role": "user", "content": prompt}] for prompt in prompt_batch
        ]
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
            for i, response in enumerate(batch_responses):
                llm_response = response.choices[0].message.content
                updated_results[batch_idx * batch_size + i]["Refined_Rationale"] = (
                    "\n".join(llm_response.splitlines()[1:])
                )
                # updated_results[batch_idx * batch_size + i][
                #     "Refined_Rationale"
                # ] = llm_response
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed completely: {str(e)}")
            continue

    final_results.extend(updated_results)

    return pd.DataFrame(final_results)
