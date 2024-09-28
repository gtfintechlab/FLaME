from pathlib import Path
PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_DIR / 'data'
LOG_DIR = PACKAGE_DIR / 'logs'
RESULTS_DIR = PACKAGE_DIR / 'results'
OUTPUT_DIR = DATA_DIR / 'outputs'
print(PACKAGE_DIR)