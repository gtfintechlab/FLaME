import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from superflue.config import LOG_LEVEL


def setup_logger(name, log_file, level=LOG_LEVEL):
    """Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Only configure the logger if it hasn't been configured before
    if not logger.handlers:
        # Set logging level
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Ensure log directory exists
        log_dir = Path(log_file).parent
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False
    
    return logger
