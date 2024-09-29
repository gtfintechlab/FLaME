import yaml
import argparse
from dotenv import load_dotenv
import os
from superflue.together_code.inference import main as inference
from huggingface_hub import login

def parse_arguments():
    parser = argparse.ArgumentParser(description="SuperFLUE")
    parser.add_argument("--config", type=str, help="Path to the YAML config file.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use.")
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
        default="superflue",
        help="Version of the prompt to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Load environment variables first
    load_dotenv()

    # Optional: Verify that environment variables are loaded
    print(f"TOGETHER_API_KEY: {os.getenv('TOGETHER_API_KEY')}")
    print(f"HUGGINGFACEHUB_API_TOKEN: {os.getenv('HUGGINGFACEHUB_API_TOKEN')}")
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    # Log in to Hugging Face if the token is set
    if HUGGINGFACEHUB_API_TOKEN:
        login(token=HUGGINGFACEHUB_API_TOKEN)
    else:
        print("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in the environment.")

    # Now import the inference function0
    args = parse_arguments()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    for key, value in config.items():
        setattr(args, key, value)

    inference(args)
