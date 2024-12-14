"""Configuration settings for the Superflue project."""

from pathlib import Path

# # Suppress all warnings related to Together.ai and function calling immediately
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings('ignore', message='.*together.*', category=Warning)
# warnings.filterwarnings('ignore', message='.*function.*calling.*', category=Warning)
# warnings.filterwarnings('ignore', message='.*response format.*', category=Warning)

# TODO: double checkif all the dir setups are needed or if there's just a bunch of dupe crap here

# Base directories
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "src" / "superflue"
DATA_DIR = ROOT_DIR / "data"

# Output directories
OUTPUT_DIR = ROOT_DIR / "output"  # Parent directory for all outputs
RESULTS_DIR = ROOT_DIR / "results"
EVALUATION_DIR = ROOT_DIR / "evaluation"
LOG_DIR = ROOT_DIR / "logs"

for directory in [DATA_DIR, OUTPUT_DIR, RESULTS_DIR, EVALUATION_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
