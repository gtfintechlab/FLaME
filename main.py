import yaml
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
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to use")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature to use"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p to use")
    parser.add_argument("--top_k", type=float, help="Top-k to use")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Inference batch size"
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "few_shot"],
        help="Version of the prompt to use",
    )
    parser.add_argument("--tasks", type=str, nargs="+", help="List of task names to run (e.g. numclaim fpb)")
    return parser.parse_args()


def run_tasks(tasks: list[str], mode: str, args):
    """Sequentially run each task by setting args.task and invoking the correct mode."""
    for t in tasks:
        args.task = t
        if mode == "inference":
            inference(args)
        else:
            evaluate(args)


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

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    for key, value in config.items():
        setattr(args, key, value)

    defaults = {
        "temperature": 0.0,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_tokens": 128,
        "batch_size": 10,
        "prompt_format": "flame",
    }

    args2 = parse_arguments()
    for key, value in vars(args2).items():
        if value and (key not in defaults or defaults.get(key) != value):
            setattr(args, key, value)

    if not args.mode or args.mode not in ["inference", "evaluate"]:
        raise ValueError(
            "Mode is required and must be either 'inference' or 'evaluate'."
        )
    if args.mode == "evaluate" and not args.file_name:
        raise ValueError("File name is required for evaluation mode.")

    tasks = args.tasks
    if not tasks:
        raise ValueError("No tasks specified; use --tasks option")
    for t in tasks:
        if t not in supported_tasks(args.mode):
            raise ValueError(f"Task '{t}' not supported for mode {args.mode}")
    run_tasks(tasks, args.mode, args)
