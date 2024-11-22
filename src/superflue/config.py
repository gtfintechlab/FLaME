import logging
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"
EVALUATION_DIR = ROOT_DIR / "evaluation_results"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_LEVEL = logging.INFO
