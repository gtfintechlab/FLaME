from flame.task_registry import EVALUATE_MAP
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.logging_utils import setup_logger

logger = setup_logger(
    name="together_evaluate",
    log_file=LOG_DIR / "together_evaluate.log",
    level=LOG_LEVEL,
)


def main(args):
    """Run evaluation for the specified task.

    Args:
        args: Command line arguments containing:
            - task: Name of the task to run
            - model: Model to use
            - file_name: Path to inference results
            - Other task-specific parameters
    """
    # support legacy args.dataset for tests, prefer args.task
    raw = getattr(args, "task", None) or getattr(args, "dataset", None)
    if not raw:
        logger.error("No task specified in args")
        return
    task = raw

    # Map of tasks to their evaluation functions
    task_evaluate_map = EVALUATE_MAP

    if task in task_evaluate_map:
        evaluate_function = task_evaluate_map[task]

        # Run evaluation
        df, metrics_df = evaluate_function(args.file_name, args)

        # Determine output base directory - import at runtime to get patched values
        from flame.config import EVALUATION_DIR, TEST_OUTPUT_DIR, IN_PYTEST

        output_dir = TEST_OUTPUT_DIR if IN_PYTEST else EVALUATION_DIR

        # Create task-specific directory
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save evaluation results
        results_filename = f"evaluation_{args.file_name.split('/')[-1]}"
        results_path = task_dir / results_filename
        df.to_csv(results_path, index=False)
        logger.info(f"Evaluation completed for {task}. Results saved to {results_path}")

        # Save metrics
        metrics_path = task_dir / f"{results_filename[:-4]}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")
    else:
        logger.error(f"Task '{task}' not found in the task evaluation map.")
