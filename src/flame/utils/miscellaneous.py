"""Miscellaneous utility functions for the FLaME framework."""

import uuid
from datetime import datetime

from flame.config import RESULTS_DIR, TEST_OUTPUT_DIR, IN_PYTEST


def generate_inference_filename(task: str, model: str, output_dir=None):
    """Generate a unique filename for inference results.

    This standardized function creates consistent filenames for all tasks
    with built-in collision prevention using timestamps and UUIDs.

    Args:
        task: The task name (e.g., 'fomc')
        model: The full model path (e.g., 'together_ai/model_name')
        output_dir: Optional output directory (defaults to RESULTS_DIR or TEST_OUTPUT_DIR)

    Returns:
        Path object for the full output file path
    """
    # Use test output directory if running in pytest, or RESULTS_DIR by default
    if output_dir is None:
        output_dir = TEST_OUTPUT_DIR if IN_PYTEST else RESULTS_DIR

    # Sanitize the model name for use in filenames
    model_parts = model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1].replace("-", "_")

    # Add timestamp and unique identifier to prevent collisions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity

    # Create the full path with unique identifiers
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{task}_{provider}_{model_name}_{timestamp}_{uid}.csv"
    return task_dir / filename
