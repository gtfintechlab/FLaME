import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
from superflue.config import LOG_DIR, DATA_DIR, ROOT_DIR, OUTPUT_DIR, PACKAGE_DIR
logger.debug(f'ROOT_DIR = {ROOT_DIR}')
logger.debug(f'DATA_DIR = {DATA_DIR}')
logger.debug(f'OUTPUT_DIR = {OUTPUT_DIR}')
logger.debug(f'LOG_DIR = {LOG_DIR}')
logger.debug(f'PACKAGE_DIR = {PACKAGE_DIR}')