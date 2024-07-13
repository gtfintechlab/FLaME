import argparse
import re
import numpy as np
import pandas as pd
from time import time
from datetime import date
from pathlib import Path
from fpb.fpb_inference import fpb_inference
from numclaim.numclaim_inference import numclaim_inference
from fomc.fomc_inference import fomc_inference
from finbench.finbench_inference import finbench_inference
from finer.finer_inference import finer_inference
from finentity.finentity_inference import finentity_inference
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a LLM on TogetherAI over the SuperFLUE dataset")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--task", type=str, help="Task to use")
    parser.add_argument("--api_key", type=str, help="API key to use")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token to use")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature to use")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p to use")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k to use")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty to use")
    parser.add_argument("--prompt_format", type=str, default="superflue", help="Version of the prompt to use")
    return parser.parse_args()

def main():
    args = parse_arguments()
    task = args.task.strip('“”"')
    
    task_inference_map = {
        "numclaim": numclaim_inference,
        "fpb": fpb_inference,
        "fomc": fomc_inference,
        "finbench": finbench_inference,
        "finer": finer_inference,
        "finentity": finentity_inference,
    }

    if task in task_inference_map:
        start_t = time()
        inference_function = task_inference_map[task]
        df = inference_function(args)
        time_taken = time() - start_t
        logger.info(f"Time taken: {time_taken:.2f} seconds")
        results_path = (
            ROOT_DIR
            / "results"
            / task
            / f"{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")

    else:
        print(f"Task '{task}' not found in the task generation map.")


if __name__ == "__main__":
    main()
