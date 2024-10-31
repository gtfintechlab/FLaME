from superflue.config import (
    LOG_DIR,
    DATA_DIR,
    ROOT_DIR,
    OUTPUT_DIR,
    PACKAGE_DIR,
    LOG_LEVEL,
)
from superflue.utils.logging_utils import setup_logger

logger = setup_logger(
    name=__name__, log_file=LOG_DIR / "superflue.log", level=LOG_LEVEL
)


logger.debug(f"ROOT_DIR = {ROOT_DIR}")
logger.debug(f"DATA_DIR = {DATA_DIR}")
logger.debug(f"OUTPUT_DIR = {OUTPUT_DIR}")
logger.debug(f"LOG_DIR = {LOG_DIR}")
logger.debug(f"PACKAGE_DIR = {PACKAGE_DIR}")
