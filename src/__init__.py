from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


ROOT_DIR = ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
SRC_DIRECTORY = ROOT_DIRECTORY / "src"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

logger.debug(f'SRC_DIRECTORY = {SRC_DIRECTORY}')
logger.debug(f'DATA_DIRECTORY = {DATA_DIRECTORY}')
