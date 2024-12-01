import time
from datetime import date

import pandas as pd
from datasets import load_dataset
from together import Together
from tqdm import tqdm

from ferrari.config import LOG_DIR, LOG_LEVEL, RESULTS_DIR
from ferrari.together_code.prompts import fincausal_task1_prompt, fincausal_task2_prompt
from ferrari.together_code.tokens import tokens
from ferrari.utils.logging_utils import setup_logger

logger = setup_logger(
    name="fincausal_inference",
    log_file=LOG_DIR / "fincausal_inference.log",
    level=LOG_LEVEL,
)


import re

import pandas as pd


def parse_fincausal_task2_response(response: str):
    cause_match = re.search(r"Cause:\s*(.*)", response)
    effect_match = re.search(r"Effect:\s*(.*)", response)
    explanation_match = re.search(r"Explanation:\s*(.*)", response)

    cause = cause_match.group(1).strip() if cause_match else "N/A"
    effect = effect_match.group(1).strip() if effect_match else "N/A"

    return cause, effect


def fincausal_inference_task1(args):
    today = date.today()
    logger.info(f"Starting FinCausal Task1 inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/FinCausal2020_Task1", trust_remote_code=True)[
        "evaluation"
    ]

    # Initialize Together API client
    client = Together()

    responses_ordered_importance = []
    for i in tqdm(range(len(dataset)), desc="Accessing FinCausal Task 1"):
        row = dataset[i]
        question = row["Text"]

        prompt = fincausal_task1_prompt(question)
        try:
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            llm_response = model_response.choices[0].message.content
            label = llm_response.splitlines()[0]
        except Exception as e:
            label = "Error"
            llm_response = f"Error: {str(e)}"

        row["llm_responses"] = label
        row["llm_complete_responses"] = llm_response
        responses_ordered_importance.append(row)

    output_df = pd.DataFrame(responses_ordered_importance)
    results_path = (
        RESULTS_DIR
        / "fincausal_task1/fincausal_task1_meta-llama-3.1-8b/"
        / f"{'fincausal_task1'}_{'llama-3.1-8b'}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    output_df.to_csv(results_path, index=False)

    logger.info(f"Inference completed. Results saved to {results_path}")
    return output_df


def fincausal_inference_task2(args):
    today = date.today()
    logger.info(f"Starting FinCausal Task2 inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/FinCausal2020_Task2", trust_remote_code=True)[
        "evaluation"
    ]
    client = Together()

    responses_ordered_importance = []
    for i in tqdm(range(len(dataset)), desc="Accessing FinCausal Task 2"):
        row = dataset[i]
        question = row["Text"]

        prompt = fincausal_task2_prompt(question)
        try:
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            llm_response = model_response.choices[0].message.content
            cause, effect = parse_fincausal_task2_response(llm_response)
        except Exception as e:
            cause = "None"
            effect = "None"
            llm_response = f"Error: {str(e)}"

        row["Cause"] = cause
        row["Effect"] = effect
        row["llm_complete_responses"] = llm_response
        responses_ordered_importance.append(row)

    output_df = pd.DataFrame(responses_ordered_importance)
    results_path = (
        RESULTS_DIR
        / "fincausal_task2/fincausal_task2_meta-llama-3.1-8b/"
        / f"{'fincausal_task2'}_{'llama-3.1-8b'}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    output_df.to_csv(results_path, index=False)

    logger.info(f"Inference completed. Results saved to {results_path}")
    return output_df
