"""Configuration settings for the Superflue project."""

import logging
from pathlib import Path
import os

# Base directories
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "src" / "superflue"
DATA_DIR = ROOT_DIR / "data"

# Output directories
OUTPUT_DIR = ROOT_DIR / "output"  # Parent directory for all outputs
RESULTS_DIR = ROOT_DIR / "results"
EVALUATION_DIR = ROOT_DIR / "evaluation"
LOG_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, RESULTS_DIR, EVALUATION_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging settings
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LITELLM_LOG_LEVEL = os.getenv("LITELLM_LOG_LEVEL", "WARNING")


def configure_logging():
    """Configure logging for the application."""
    # Configure root logger
    logging.basicConfig(level=LOG_LEVEL)

    # Configure LiteLLM logging
    litellm_logger = logging.getLogger("litellm")
    litellm_logger.setLevel(LITELLM_LOG_LEVEL)

    # Add file handler for LiteLLM logs if we want to capture them
    if LITELLM_LOG_LEVEL != "ERROR":  # Only capture if not ERROR level
        litellm_handler = logging.FileHandler(LOG_DIR / "litellm.log")
        litellm_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        litellm_logger.addHandler(litellm_handler)

    # Configure OpenAI logging (used by LiteLLM)
    openai_logger = logging.getLogger("openai")
    openai_logger.setLevel(LITELLM_LOG_LEVEL)
