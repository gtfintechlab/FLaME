"""Configuration settings for the Ferrari project."""

import logging
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "src" / "ferrari"
DATA_DIR = ROOT_DIR / "data"

# Output directories
OUTPUT_DIR = ROOT_DIR / "output"  # Parent directory for all outputs
RESULTS_DIR = OUTPUT_DIR / "results"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
LOG_DIR = OUTPUT_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, RESULTS_DIR, EVALUATION_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging settings
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"