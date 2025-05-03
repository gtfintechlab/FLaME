import yaml
import argparse
from dotenv import load_dotenv
import os
from flame.code.inference import main as inference
from huggingface_hub import login
from flame.code.evaluate import main as evaluate


def parse_arguments():
    parser = argparse.ArgumentParser(description="FLaME")
    parser.add_argument("--config", type=str, help="Path to the YAML config file.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use.")
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
        default="flame",
        help="Version of the prompt to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Load environment variables first
    load_dotenv()

    # Optional: Verify that environment variables are loaded
    print(f"TOGETHER_API_KEY: {os.getenv('TOGETHERAI_API_KEY')}")
    print(f"HUGGINGFACEHUB_API_TOKEN: {os.getenv('HUGGINGFACEHUB_API_TOKEN')}")
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # Log in to Hugging Face if the token is set
    if HUGGINGFACEHUB_API_TOKEN:
        login(token=HUGGINGFACEHUB_API_TOKEN)
    else:
        print(
            "Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in the environment."
        )

    # Now import the inference function
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

    if args.mode == "inference":
        inference(args)
    elif args.mode == "evaluate":
        evaluate(args)
