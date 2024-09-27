import argparse
import logging
from datetime import date
from pathlib import Path
from time import time
import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from banking77.banking77_inference import banking77_inference
from finbench.finbench_inference import finbench_inference
from finentity.finentity_inference import finentity_inference
from finer.finer_inference import finer_inference
from fomc.fomc_inference import fomc_inference
from fpb.fpb_inference import fpb_inference
from numclaim.numclaim_inference import numclaim_inference
from causal_classification.causal_classification_inference import causal_classification_inference 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils.api_utils import make_api_call, save_raw_output
from src.utils.logging_utils import setup_logger

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "data" / "outputs"
LOG_DIR = ROOT_DIR / "logs"
logging.basicConfig(level=logging.INFO)
logger = setup_logger("main_inference", LOG_DIR / "main_inference.log")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a LLM on TogetherAI over the SuperFLUE dataset"
    )
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--task", type=str, help="Task to use")
    parser.add_argument("--api_key", type=str, help="API key to use")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token to use")
    
    # Load config.yaml
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        parser.add_argument(
            "--max_tokens",
            type=int,
            default=config["fpb"]["max_tokens"],
            help="Max tokens to use",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=config["fpb"]["temperature"],
            help="Temperature to use",
        )
        parser.add_argument(
            "--top_p", type=float, default=config["fpb"]["top_p"], help="Top-p to use"
        )
        parser.add_argument(
            "--top_k", type=int, default=config["fpb"]["top_k"], help="Top-k to use"
        )
        parser.add_argument(
            "--repetition_penalty",
            type=float,
            default=config["fpb"]["repetition_penalty"],
            help="Repetition penalty to use",
        )
        parser.add_argument(
            "--prompt_format",
            type=str,
            default="superflue",
            help="Version of the prompt to use",
        )
    
    return parser.parse_args()



def process_api_response(results, task, model):
    save_raw_output(results, task, model, OUTPUT_DIR)
    return results


def main():
    args = parse_arguments()
    task = args.task.strip('"')
    task = args.task.strip('""')

    task_inference_map = {
        "numclaim": numclaim_inference,
        "fpb": fpb_inference,
        "fomc": fomc_inference,
        "finbench": finbench_inference,
        "finer": finer_inference,
        "finentity": finentity_inference,
        "banking77": banking77_inference,
        "causal_classification": causal_classification_inference,
    }

    if task in task_inference_map:
        start_t = time()
        inference_function = task_inference_map[task]
        df = inference_function(args, make_api_call, process_api_response)
        time_taken = time() - start_t
        logger.info(f"Time taken: {time_taken:.2f} seconds")
        results_path = (
            OUTPUT_DIR
            / task
            / args.model
            / f"{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Inference completed for {task}. Results saved to {results_path}")
        # metrics = evaluate(df, 'response', task_regex[task], task_label_mapping[task])
        # print(metrics)

    else:
        logger.error(f"Task '{task}' not found in the task generation map.")


if __name__ == "__main__":
    main()
