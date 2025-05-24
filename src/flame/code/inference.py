from time import time

from flame.task_registry import INFERENCE_MAP
from flame.utils.logging_utils import get_component_logger
from flame.utils.output_utils import generate_output_path

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference")


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

        # Generate a unique results path for this run using new structure
        from flame.config import RESULTS_DIR, TEST_OUTPUT_DIR, IN_PYTEST

        base_dir = TEST_OUTPUT_DIR if IN_PYTEST else RESULTS_DIR
        results_path = generate_output_path(base_dir, task, args.model)

        df.to_csv(results_path, index=False)
        logger.info(f"Inference completed for {task}. Results saved to {results_path}")
    else:
        logger.error(f"Task '{task}' not found in the task generation map.")
