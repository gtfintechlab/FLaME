# import logging
import time
from datetime import date

# from pathlib import Path
from litellm import completion 

import nltk
import pandas as pd
from datasets import load_dataset

# Mock imports for custom TATQA prompt and tokens
from superflue.together_code.prompts import (
    tatqa_prompt,
)  # To be implemented for TAT-QA prompt
from superflue.together_code.tokens import tokens  # Token logic for TAT-QA

nltk.download("punkt")


from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="tatqa_inference", log_file=LOG_DIR / "tatqa_inference.log", level=LOG_LEVEL
)


def tatqa_inference(args):
    today = date.today()
    logger.info(f"Starting TAT-QA inference on {today}")

    logger.info("Loading dataset...")
    # Replace with the appropriate Hugging Face dataset for TAT-QA
    dataset = load_dataset("gtfintechlab/TATQA", trust_remote_code=True)

    # Initialize lists to store the question, context, actual answers, and model responses
    questions = []
    contexts = []
    actual_answers = []
    model_responses = []

    logger.info(f"Starting inference on {args.task}...")
    # start_t = time.time()
    for i in range(len(dataset["test"])): # type: ignore
        question = dataset["test"][i]["query"] # type: ignore
        context = dataset["test"][i]["text"] # type: ignore
        actual_answer = dataset["test"][i]["answer"] # type: ignore

        questions.append(question)
        contexts.append(context)
        actual_answers.append(actual_answer)

        try:
            logger.info(f"Processing question {i+1}/{len(dataset['test'])}") # type: ignore
            # TAT-QA-specific prompt logic, create the prompt for table and text-based QA
            model_response = completion(
                prompt=tatqa_prompt(question, context),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            model_responses.append(model_response.choices[0].message.content)

            df = pd.DataFrame(
                {
                    "questions": questions,
                    "contexts": contexts,
                    "actual_answers": actual_answers,
                    "model_responses": model_responses,
                }
            )

        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}")
            time.sleep(20.0)
            continue

    results_path = (
        RESULTS_DIR
        / args.task
        / f"{args.task}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
