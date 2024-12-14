from superflue.config import (
    LOG_DIR,
    DATA_DIR,
    ROOT_DIR,
    OUTPUT_DIR,
    PACKAGE_DIR,
)

from superflue.utils.logging_utils import get_logger

logger = get_logger(__name__)

logger.debug(f"ROOT_DIR = {ROOT_DIR}")
logger.debug(f"DATA_DIR = {DATA_DIR}")
logger.debug(f"OUTPUT_DIR = {OUTPUT_DIR}")
logger.debug(f"LOG_DIR = {LOG_DIR}")
logger.debug(f"PACKAGE_DIR = {PACKAGE_DIR}")
