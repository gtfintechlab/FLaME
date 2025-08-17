#!/usr/bin/env python3
"""Debug script to test BenchForge integration."""

import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test BenchForge main
try:
    from argparse import Namespace

    # Create minimal args for testing
    args = Namespace(
        mode="inference",
        task="fomc",
        dataset=None,
        model="together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        max_tokens=20,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        batch_size=5,
        prompt_format="zero_shot",
        num_samples=5,
        split="test",
        seed=42,
        file_name=None,
        metrics=None,
        output_dir=None,
        results_dir=None,
        evaluation_dir=None,
        config=None,  # This might be the issue
        verbose=True,
        quiet=False,
    )

    # Try to run inference directly
    from flame.main_benchforge import run_inference

    logger.info("Testing BenchForge inference with minimal args...")
    logger.info(f"Args: {args}")

    # Check if config path is causing the issue
    if hasattr(args, "config") and args.config is None:
        logger.warning("Config is None - this might be the issue")

    # Try running
    run_inference(args)

except Exception as e:
    logger.error(f"Error occurred: {e}", exc_info=True)
    import traceback

    traceback.print_exc()

    # Check the specific error
    if "expected str, bytes or os.PathLike object, not NoneType" in str(e):
        logger.error("This is the NoneType path error!")
        logger.error("Checking which path is None...")
