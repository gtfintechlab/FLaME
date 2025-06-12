from flame.config import (
    DATA_DIR,
    LOG_DIR,
    LOG_LEVEL,
    PACKAGE_DIR,
    ROOT_DIR,
)
from flame.utils.logging_utils import setup_logger

logger = setup_logger(name=__name__, log_file=LOG_DIR / "flame.log", level=LOG_LEVEL)


logger.debug(f"ROOT_DIR = {ROOT_DIR}")
logger.debug(f"DATA_DIR = {DATA_DIR}")
logger.debug(f"LOG_DIR = {LOG_DIR}")
logger.debug(f"PACKAGE_DIR = {PACKAGE_DIR}")
