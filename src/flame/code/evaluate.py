from flame.task_registry import EVALUATE_MAP
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.logging_utils import setup_logger
from flame.utils.output_utils import generate_output_path, parse_output_filename

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

        base_dir = TEST_OUTPUT_DIR if IN_PYTEST else EVALUATION_DIR

        # Extract model info from input filename to maintain consistency
        input_filename = args.file_name.split("/")[-1]

        try:
            # Try to parse the new format filename
            parsed = parse_output_filename(input_filename)
            # Use the model from args if available, otherwise reconstruct from parsed filename
            if hasattr(args, "model") and args.model:
                model_info = args.model
            else:
                # Extract provider and model family from the parsed model_slug
                # model_slug format: "llama-4-scout-17b-16e-instruct"
                model_slug = parsed["model_slug"]
                # For now, we'll use the model info from args or fall back to a reasonable default
                model_info = f"together_ai/meta-llama/{model_slug}"
            run = parsed["run"]
        except (ValueError, KeyError):
            # Fall back to legacy parsing if new format fails
            # Legacy format: {task}_{provider}_{model_name}_{timestamp}_{uid}.csv
            parts = input_filename.replace(".csv", "").split("_")
            if len(parts) >= 3:
                model_info = f"{parts[1]}/{parts[2]}"  # provider/model
                run = 1
            else:
                model_info = "unknown/unknown"
                run = 1

        # Generate evaluation results path using new structure
        results_path = generate_output_path(
            base_dir, task, model_info, run, metrics=False
        )
        df.to_csv(results_path, index=False)
        logger.info(f"Evaluation completed for {task}. Results saved to {results_path}")

        # Generate metrics path using new structure
        metrics_path = generate_output_path(
            base_dir, task, model_info, run, metrics=True
        )
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")
    else:
        logger.error(f"Task '{task}' not found in the task evaluation map.")
