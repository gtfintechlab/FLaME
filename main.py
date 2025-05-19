import argparse
from dotenv import load_dotenv
import os
from flame.code.inference import main as inference
from huggingface_hub import login
from flame.code.evaluate import main as evaluate
from flame.task_registry import supported as supported_tasks


def parse_arguments():
    parser = argparse.ArgumentParser(description="FLaME")
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
        "--sample_size",
        type=int,
        help="Number of samples to use from each dataset (defaults to all samples)",
    )
    args = parser.parse_args()

    # Load and merge config
    config = {}
    if args.config:
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
    # CLI overrides config; fill missing from config
    for key, value in config.items():
        if getattr(args, key, None) is None:
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
        "sample_size": None,  # None means use all samples
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


def run_tasks(tasks: list[str], mode: str, args):
    # validate supported tasks
    supported = supported_tasks(mode)
    for t in tasks:
        if t not in supported:
            raise ValueError(f"Task '{t}' not supported for mode {mode}")
    """Sequentially run each task, collect errors, and raise MultiTaskError if any fail."""
    errors: dict[str, Exception] = {}
    for t in tasks:
        args.task = t
        try:
            if mode == "inference":
                inference(args)
            else:
                evaluate(args)
        except Exception as e:
            errors[t] = e
    if errors:
        raise MultiTaskError(errors)


if __name__ == "__main__":
    load_dotenv()
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if HUGGINGFACEHUB_API_TOKEN:
        login(token=HUGGINGFACEHUB_API_TOKEN)
    else:
        print(
            "Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in the environment."
        )

    args = parse_arguments()

    if not args.mode or args.mode not in ["inference", "evaluate"]:
        raise ValueError(
            "Mode is required and must be either 'inference' or 'evaluate'."
        )
    if args.mode == "evaluate" and not args.file_name:
        raise ValueError("File name is required for evaluation mode.")

    tasks = args.tasks
    if not tasks:
        raise ValueError("No tasks specified; use --tasks option")
    run_tasks(tasks, args.mode, args)
