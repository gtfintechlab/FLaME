"""Argument parsing module."""

import argparse
import os


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SuperFLUE tasks")

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["banking77", "fomc", "fpb"],
        help="Task to run",
    )

    # Model configuration
    parser.add_argument(
        "--inference_model",
        type=str,
        required=True,
        help="Model to use for inference",
    )
    parser.add_argument(
        "--extraction_model",
        type=str,
        help="Model to use for extraction (if needed)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_org",
        type=str,
        default="superflue",
        help="Organization name for dataset",
    )

    # Inference parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for inference",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top p for inference",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top k for inference",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
        help="Maximum tokens for inference",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for inference",
    )

    # Logging configuration
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--litellm_log_level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="LiteLLM logging level",
    )

    args = parser.parse_args()

    # Set logging levels from args
    os.environ["LOG_LEVEL"] = args.log_level
    os.environ["LITELLM_LOG_LEVEL"] = args.litellm_log_level

    return args
