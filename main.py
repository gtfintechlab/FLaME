"""Main entry point for SuperFLUE."""

import os
import warnings
import logging
from pathlib import Path

# Configure warnings before any imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*together.*function.*calling.*")
warnings.filterwarnings("ignore", message=".*together.*", category=Warning)
warnings.filterwarnings("ignore", message=".*function.*calling.*", category=Warning)
warnings.filterwarnings("ignore", message=".*response format.*", category=Warning)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "src" / "superflue"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
RESULTS_DIR = ROOT_DIR / "results"
EVALUATION_DIR = ROOT_DIR / "evaluation"
LOG_DIR = ROOT_DIR / "logs"

for directory in [DATA_DIR, OUTPUT_DIR, RESULTS_DIR, EVALUATION_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load other imports after warning configuration
import yaml
import argparse
from dotenv import load_dotenv
from huggingface_hub import login
from superflue.utils.logging_utils import configure_root_logger
from superflue.config import LOG_DIR


def configure_env_from_args(args):
    """Configure environment variables from command line args."""
    if hasattr(args, "log_level"):
        os.environ["LOG_LEVEL"] = args.log_level.upper()
    if hasattr(args, "litellm_log_level"):
        os.environ["LITELLM_LOG"] = args.litellm_log_level.upper()


def configure_env_from_config(config):
    """Configure environment variables from config file."""
    if "log_level" in config:
        os.environ.setdefault("LOG_LEVEL", config["log_level"].upper())
    if "litellm_log_level" in config:
        os.environ.setdefault("LITELLM_LOG", config["litellm_log_level"].upper())


def parse_arguments():
    parser = argparse.ArgumentParser(description="SuperFLUE")

    # Core arguments
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

    # Model arguments
    parser.add_argument(
        "--inference-model", type=str, help="Model to use for inference"
    )
    parser.add_argument(
        "--extraction-model", type=str, help="Model to use for evaluation/extraction"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="[DEPRECATED] Use --inference-model or --extraction-model instead",
    )

    # Inference parameters
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

    # Logging configuration
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    return parser.parse_args()


def main():
    # Load environment variables first
    load_dotenv()

    # Parse command line arguments
    args = parse_arguments()

    # Load config file if specified
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Configure environment before any other imports
    configure_env_from_config(config)  # Config file provides defaults
    configure_env_from_args(args)  # Command line args override config

    # Configure logging ONCE for the entire application
    configure_root_logger(LOG_DIR, args=args)
    logger = logging.getLogger(__name__)

    # Handle model configuration from config
    if "models" in config:
        if not args.inference_model:
            args.inference_model = config["models"].get("inference")
        if not args.extraction_model:
            args.extraction_model = config["models"].get("extraction")

    # Handle deprecated --model argument
    if args.model:
        logger.warning(
            "--model argument is deprecated. Use --inference-model or --extraction-model instead"
        )
        if not args.inference_model and not args.extraction_model:
            args.inference_model = args.model
            args.extraction_model = args.model

    # Set other config values
    for key, value in config.items():
        if key not in {
            "models",
            "log_level",
            "litellm_log_level",
        }:  # Skip already handled keys
            setattr(args, key, value)

    # Validate arguments
    if not args.mode or args.mode not in ["inference", "evaluate"]:
        logger.error("Mode is required and must be either 'inference' or 'evaluate'.")
        raise ValueError(
            "Mode is required and must be either 'inference' or 'evaluate'."
        )

    if args.mode == "evaluate":
        if not args.file_name:
            logger.error("File name is required for evaluation mode.")
            raise ValueError("File name is required for evaluation mode.")
        if not args.extraction_model:
            logger.error("Extraction model is required for evaluation mode.")
            raise ValueError("Extraction model is required for evaluation mode.")

    if args.mode == "inference" and not args.inference_model:
        logger.error("Inference model is required for inference mode.")
        raise ValueError("Inference model is required for inference mode.")

    # Now it's safe to import modules that depend on logging configuration
    from superflue.code.inference import main as inference
    from superflue.code.evaluate import main as evaluate

    # Login to Hugging Face if token is available
    if os.getenv("HUGGINGFACE_TOKEN"):
        login(token=os.getenv("HUGGINGFACE_TOKEN"))

    # Run the appropriate mode
    if args.mode == "inference":
        inference(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        logger.error(f"Invalid mode: {args.mode}")
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
