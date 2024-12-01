import argparse
from datetime import date
from time import time

from ferrari.config import LOG_DIR, LOG_LEVEL, RESULTS_DIR
from ferrari.together_code.causal_classification.causal_classification_inference import (
    causal_classification_inference,
)
from ferrari.together_code.econlogicqa.econlogicqa_inference import (
    econlogicqa_inference,
)
from ferrari.together_code.finbench.finbench_inference import finbench_inference
from ferrari.together_code.bizbench.bizbench_inference import bizbench_inference
from ferrari.together_code.finer.finer_inference import finer_inference
from ferrari.together_code.finentity.finentity_inference import finentity_inference
from ferrari.together_code.headlines.headlines_inference import headlines_inference
from ferrari.together_code.fiqa.fiqa_task1_inference import fiqa_inference
from ferrari.together_code.fiqa.fiqa_task2_inference import fiqa_task2_inference
from ferrari.together_code.fomc.fomc_inference import fomc_inference
from ferrari.together_code.fpb.fpb_inference import fpb_inference
from ferrari.utils.logging_utils import setup_logger

logger = setup_logger(
    name="together_inference",
    log_file=LOG_DIR / "together_inference.log",
    level=LOG_LEVEL,
)


def main(args):
    task = args.dataset.strip('“”"')
    task_inference_map = {
        "fpb": fpb_inference,
        "fomc": fomc_inference,
        "finbench": finbench_inference,
        "bizbench": bizbench_inference,
        "finer": finer_inference,
        "finentity": finentity_inference,
        "headlines": headlines_inference,
        "fiqa_task1": fiqa_inference,
        "fiqa_task2": fiqa_task2_inference,
        "hg_fiqa_task1": hg_fiqa_inference,
    }

    if task in task_inference_map:
        start_t = time()
        inference_function = task_inference_map[task]
        df = inference_function(args)
        time_taken = time() - start_t
        logger.info(f"Time taken for inference: {time_taken}")
        results_path = (
            RESULTS_DIR
            / task
            / f"{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Inference completed for {task}. Results saved to {results_path}")
    else:
        logger.error(f"Task '{task}' not found in the task generation map.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Dataset to run inference on")
    parser.add_argument("--model", type=str, help="Model to run inference with")
    parser.add_argument("--max_tokens", type=int, help="Max tokens for model inference")
    parser.add_argument(
        "--temperature", type=float, help="Temperature for model inference"
    )
    parser.add_argument("--top_k", type=int, help="Top k for model inference")
    parser.add_argument("--top_p", type=float, help="Top p for model inference")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        help="Repetition penalty for model inference",
    )
    args = parser.parse_args()
    main(args)
