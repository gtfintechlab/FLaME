import argparse
import logging
from datetime import date
from pathlib import Path
from time import time

# Only import fiqa_task1_inference since you're testing FiQA
from fiqa.fiqa_task1_inference import fiqa_inference

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

    # Directly assign defaults for FiQA task
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature to use",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.7, help="Top-p to use"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-k to use"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
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

    task_inference_map = {
        "fiqa_task1": fiqa_inference  
    }

    if task in task_inference_map:
        start_t = time()
        inference_function = task_inference_map[task]
        df = inference_function(args)
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
    else:
        logger.error(f"Task '{task}' not found in the task generation map.")


if __name__ == "__main__":
    main()
