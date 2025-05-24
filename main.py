import argparse
from dotenv import load_dotenv
import os
import sys
import logging
from flame.code.inference import main as inference
from huggingface_hub import login
from flame.code.evaluate import main as evaluate
from flame.task_registry import supported as supported_tasks
from flame.config import configure_logging, LOG_CONFIG
from flame.utils.logging_utils import get_component_logger
import litellm


def parse_arguments():
    parser = argparse.ArgumentParser(description="FLaME")

    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add list-tasks command
    subparsers.add_parser("list-tasks", help="List available tasks")

    # Keep the original arguments as the default (no subcommand)
    parser.add_argument("--config", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inference", "evaluate"],
        help="Mode to run: inference or evaluate.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        help="File name for evaluation (required for mode=evaluate).",
    )
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--max_tokens", type=int, help="Max tokens to use")
    parser.add_argument("--temperature", type=float, help="Temperature to use")
    parser.add_argument("--top_p", type=float, help="Top-p to use")
    parser.add_argument("--top_k", type=float, help="Top-k to use")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        help="Repetition penalty to use",
    )
    parser.add_argument("--batch_size", type=int, help="Inference batch size")
    parser.add_argument(
        "--prompt_format",
        type=str,
        choices=["zero_shot", "few_shot"],
        help="Version of the prompt to use",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="List of task names to run (e.g. numclaim fpb)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for this run (overrides config)",
    )
    args = parser.parse_args()

    # Load and merge config
    config = {}
    if args.config:
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    # Configure logging from YAML config
    configure_logging(config)

    # Override logging level if --debug flag is used
    if args.debug:
        LOG_CONFIG["level"] = logging.DEBUG
        LOG_CONFIG["console"]["level"] = logging.DEBUG

    # CLI overrides config; fill missing from config
    for key, value in config.items():
        if key != "logging" and getattr(args, key, None) is None:
            setattr(args, key, value)

    # Apply defaults
    defaults = {
        "model": None,
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": None,
        "repetition_penalty": 1.0,
        "batch_size": 10,
        "prompt_format": "zero_shot",
    }
    for key, default in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default)

    return args


class MultiTaskError(Exception):
    """Raised when one or more tasks fail during multi-task execution."""

    def __init__(self, errors: dict[str, Exception]):
        self.errors = errors
        super().__init__(f"Errors in tasks: {', '.join(errors.keys())}")


def apply_task_specific_config(args, task: str):
    """Apply task-specific configuration overrides from YAML config.

    Args:
        args: Global arguments object
        task: Task name to apply specific config for

    Returns:
        Modified args object with task-specific overrides applied
    """
    import copy

    # Create a copy to avoid modifying the original args
    task_args = copy.deepcopy(args)

    # Apply task-specific config if available
    if hasattr(args, "task_config") and args.task_config and task in args.task_config:
        task_config = args.task_config[task]
        logger = get_component_logger("flame")
        logger.debug(f"Applying task-specific config for '{task}': {task_config}")

        for key, value in task_config.items():
            if hasattr(task_args, key):
                original_value = getattr(task_args, key)
                setattr(task_args, key, value)
                logger.debug(f"Task '{task}': {key} {original_value} -> {value}")
            else:
                # Add new task-specific parameter
                setattr(task_args, key, value)
                logger.debug(f"Task '{task}': Added new parameter {key} = {value}")

    return task_args


def run_tasks(tasks: list[str], mode: str, args):
    # Get the main logger
    logger = get_component_logger("flame")

    # validate supported tasks
    supported = supported_tasks(mode)
    for t in tasks:
        if t not in supported:
            logger.error(f"Task '{t}' not supported for mode {mode}")
            raise ValueError(f"Task '{t}' not supported for mode {mode}")

    """Sequentially run each task, collect errors, and raise MultiTaskError if any fail."""
    errors: dict[str, Exception] = {}
    for t in tasks:
        # Apply task-specific configuration
        task_args = apply_task_specific_config(args, t)
        task_args.task = t

        logger.info(f"Running task '{t}' in {mode} mode")
        try:
            if mode == "inference":
                inference(task_args)
            else:
                evaluate(task_args)
            logger.info(f"Task '{t}' completed successfully")
        except Exception as e:
            logger.error(f"Task '{t}' failed with error: {str(e)}", exc_info=True)
            errors[t] = e
    if errors:
        logger.error(
            f"Encountered errors in {len(errors)} tasks: {', '.join(errors.keys())}"
        )
        raise MultiTaskError(errors)


# Use the configure_litellm from config.py


if __name__ == "__main__":
    load_dotenv()

    # Pre-configure litellm with default settings to suppress early logging
    import litellm

    litellm.verbose = False
    litellm.set_verbose = False
    litellm.suppress_debug_info = True
    litellm.drop_params = True

    # Parse arguments and configure logging
    args = parse_arguments()

    # Handle list-tasks command
    if args.command == "list-tasks":
        print("Available inference tasks:")
        for task in sorted(supported_tasks("inference")):
            print(f"  - {task}")
        print("\nAvailable evaluation tasks:")
        for task in sorted(supported_tasks("evaluate")):
            print(f"  - {task}")
        sys.exit(0)

    # Set up main logger
    main_logger = get_component_logger("flame.main")
    main_logger.info("Starting FLaME framework")

    # Set up root logger to be quiet by default
    logging.basicConfig(level=logging.WARNING)

    # Configure litellm
    from flame.config import configure_litellm

    litellm_logger = configure_litellm()
    litellm_logger.debug("LiteLLM configured")

    # Filter out HTTP request logs by monkey-patching the logging module
    original_log = logging.Logger._log

    def filtered_log(self, level, msg, args, **kwargs):
        # Filter out HTTP request logs
        if level < logging.WARNING and isinstance(msg, str) and "HTTP Request:" in msg:
            # Don't log HTTP requests unless at WARNING level or higher
            return

        # Call the original logging function for everything else
        original_log(self, level, msg, args, **kwargs)

    # Apply the monkey patch
    logging.Logger._log = filtered_log

    # Additional suppression of HTTP request logs
    for logger_name in logging.root.manager.loggerDict:
        if any(
            term in logger_name.lower() for term in ["http", "request", "api", "llm"]
        ):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Explicitly handle known noisy loggers
    for logger_name in [
        "httpx",
        "urllib3",
        "requests",
        "httpcore",
        "litellm.llms",
        "openai",
    ]:
        if logger_name in logging.root.manager.loggerDict:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Debug diagnostic to find the source of HTTP logs if --debug is used
    if args.debug:
        main_logger.debug("Active loggers:")
        active_loggers = sorted(list(logging.root.manager.loggerDict.keys()))
        for logger_name in active_loggers:
            main_logger.debug(
                f"  - {logger_name}: {logging.getLogger(logger_name).level}"
            )

    # Log HuggingFace status
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if HUGGINGFACEHUB_API_TOKEN:
        try:
            login(token=HUGGINGFACEHUB_API_TOKEN)
            main_logger.info("Logged in to Hugging Face Hub")
        except Exception as e:
            main_logger.error(f"Failed to authenticate with Hugging Face Hub: {e}")
            print(f"ERROR: Failed to authenticate with Hugging Face Hub: {e}")
            print("Please check your HUGGINGFACEHUB_API_TOKEN is valid.")
            sys.exit(1)
    else:
        main_logger.error(
            "Hugging Face API token not found. This is required for accessing FLaME datasets."
        )
        print(
            "ERROR: Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in the environment."
        )
        print("The FLaME datasets are private and require authentication.")
        sys.exit(1)

    # Validation only needed when not running a command
    if args.command is None:
        if not args.mode or args.mode not in ["inference", "evaluate"]:
            main_logger.error(
                "Mode is required and must be either 'inference' or 'evaluate'."
            )
            raise ValueError(
                "Mode is required and must be either 'inference' or 'evaluate'."
            )
        if args.mode == "evaluate" and not args.file_name:
            main_logger.error("File name is required for evaluation mode.")
            raise ValueError("File name is required for evaluation mode.")

        tasks = args.tasks
        if not tasks:
            main_logger.error("No tasks specified; use --tasks option")
            raise ValueError("No tasks specified; use --tasks option")

        main_logger.info(
            f"Running {len(tasks)} tasks in {args.mode} mode: {', '.join(tasks)}"
        )
        run_tasks(tasks, args.mode, args)
        main_logger.info("All tasks completed successfully")
