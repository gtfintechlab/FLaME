from time import time
from datetime import date

from flame.code.fomc.fomc_inference import fomc_inference
from flame.code.finer.finer_inference import finer_inference
from flame.code.finentity.finentity_inference import finentity_inference
from flame.code.subjectiveqa.subjectiveqa_inference import subjectiveqa_inference
from flame.task_registry import INFERENCE_MAP
from flame.config import LOG_DIR, RESULTS_DIR, LOG_LEVEL
from flame.utils.logging_utils import setup_logger

logger = setup_logger(
    name="together_inference",
    log_file=LOG_DIR / "together_inference.log",
    level=LOG_LEVEL,
)


def main(args):
    """Run inference for the specified task.

    Args:
        args: Command line arguments containing:
            - task: Name of the task to run
            - model: Model to use
            - Other task-specific parameters
    """
    task = args.task.strip('"""')

    task_inference_map = INFERENCE_MAP

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
