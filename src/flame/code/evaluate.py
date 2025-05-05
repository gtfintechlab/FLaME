from flame.task_registry import EVALUATE_MAP
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.logging_utils import setup_logger
from pathlib import Path

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
    task = args.task

    # Map of tasks to their evaluation functions
    task_evaluate_map = EVALUATE_MAP

    if task in task_evaluate_map:
        evaluate_function = task_evaluate_map[task]

        # Run evaluation
        df, metrics_df = evaluate_function(args.file_name, args)

        # Save evaluation results
        results_path = f"evaluation_{args.file_name}"
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Evaluation completed for {task}. Results saved to {results_path}")

        # Save metrics
        metrics_path = Path(f"{str(results_path)[:-4]}_metrics.csv")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")
    else:
        logger.error(f"Task '{task}' not found in the task evaluation map.")
