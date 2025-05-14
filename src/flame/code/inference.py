from time import time
from datetime import date

from flame.task_registry import INFERENCE_MAP
from flame.config import LOG_DIR, RESULTS_DIR, LOG_LEVEL, TEST_OUTPUT_DIR, IN_PYTEST
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
    # support legacy args.dataset for tests, prefer args.task
    raw = getattr(args, "task", None) or getattr(args, "dataset", None)
    if not raw:
        logger.error("No task specified in args")
        return
    task = raw.strip('"""')

    task_inference_map = INFERENCE_MAP

    if task in task_inference_map:
        start_t = time()
        inference_function = task_inference_map[task]
        df = inference_function(args)
        time_taken = time() - start_t
        logger.info(f"Time taken for inference: {time_taken}")

        # Use test output directory if running in pytest
        output_dir = TEST_OUTPUT_DIR if IN_PYTEST else RESULTS_DIR

        # Create the task-specific subfolder
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        # Generate the output path
        results_path = (
            task_dir / f"{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )

        df.to_csv(results_path, index=False)
        logger.info(f"Inference completed for {task}. Results saved to {results_path}")
    else:
        logger.error(f"Task '{task}' not found in the task generation map.")
