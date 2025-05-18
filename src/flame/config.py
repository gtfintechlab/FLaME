"""Configuration settings for the FLaME project.

TEST OUTPUT PATTERN:
When IN_PYTEST is True (set by conftest.py), all outputs should use:
- TEST_OUTPUT_DIR instead of RESULTS_DIR
- TEST_OUTPUT_DIR instead of EVALUATION_DIR

This ensures test artifacts are isolated and gitignored.
"""

import logging
from pathlib import Path
import os

# Base directories
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "src" / "flame"
DATA_DIR = ROOT_DIR / "data"

# Output directories
RESULTS_DIR = ROOT_DIR / "results"
EVALUATION_DIR = ROOT_DIR / "evaluations"  # Changed to match existing directory name
LOG_DIR = ROOT_DIR / "logs"
TEST_OUTPUT_DIR = (
    ROOT_DIR / "tests" / "test_outputs"
)  # Dedicated directory for test artifacts

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RESULTS_DIR,
    EVALUATION_DIR,
    LOG_DIR,
    TEST_OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging settings
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Check if running inside pytest
IN_PYTEST = bool(os.environ.get("PYTEST_RUNNING", ""))
