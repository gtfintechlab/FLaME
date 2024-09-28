import sys
from pathlib import Path
import logging
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT_DIR / 'logs'
SRC_DIRECTORY = ROOT_DIR / 'src'
DATA_DIRECTORY = ROOT_DIR / 'data'
OUTPUT_DIR = DATA_DIRECTORY / 'outputs'
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))
    
import argparse
from time import time
from datetime import date
from fpb.fpb_inference import fpb_inference
from numclaim.numclaim_inference import numclaim_inference
from fnxl.fnxl_inference import fnxl_inference
from fomc.fomc_inference import fomc_inference
from finbench.finbench_inference import finbench_inference
from finer.finer_inference import finer_inference
from finentity.finentity_inference import finentity_inference
from headlines.headlines_inference import headlines_inference
from finqa.fiqa_task1_inference import fiqa_inference
from finqa.fiqa_task2_inference import fiqa_task2_inference
from edtsum.edtsum_inference import edtsum_inference
from src.utils.logging_utils import setup_logger
from time import time

logger = setup_logger(name="together_inference", log_file = LOG_DIR / "together_inference.log", level=logging.DEBUG)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a LLM on TogetherAI over the SuperFLUE dataset")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--task", type=str, help="Task to use")
    parser.add_argument("--api_key", type=str, help="API key to use")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token to use")

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Max tokens to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature to use",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p to use"
    )
    # parser.add_argument(
    #     "--top_k", type=int, default=50, help="Top-k to use"
    # )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty to use",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="superflue",
        help="Version of the prompt to use",
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    task = args.task.strip('“”"')

    task_inference_map = {
        'numclaim': numclaim_inference,
        'fpb': fpb_inference,
        'fomc': fomc_inference,
        'finbench': finbench_inference,
        'finer': finer_inference,
        'finentity': finentity_inference,
        'headlines': headlines_inference,
        'fiqa_task1': fiqa_inference, # double check this i think it might be _task1_
        'fiqa_task2' : fiqa_task2_inference,
        'edt_sum':edtsum_inference,
        'fnxl': fnxl_inference,
    }

    if task in task_inference_map:
        start_t = time()
        inference_function = task_inference_map[task]
        df = inference_function(args)
        time_taken = time() - start_t
        print(time_taken)
        results_path = ROOT_DIR / 'results' / task  / f"{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Inference completed for {task}. Results saved to {results_path}")
    else:
        print(f"Task '{task}' not found in the task generation map.")

if __name__ == "__main__":
    main()
