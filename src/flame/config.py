"""Configuration settings for the FLaME project."""

import logging
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "src" / "flame"
DATA_DIR = ROOT_DIR / "data"

# Output directories
OUTPUT_DIR = ROOT_DIR / "output"  # Parent directory for all outputs
RESULTS_DIR = ROOT_DIR / "results"
EVALUATION_DIR = ROOT_DIR / "evaluation"
LOG_DIR = ROOT_DIR / "logs"
TEST_OUTPUT_DIR = ROOT_DIR / "test_outputs"  # Dedicated directory for test artifacts

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, RESULTS_DIR, EVALUATION_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging settings
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
