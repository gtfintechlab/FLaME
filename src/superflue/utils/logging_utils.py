import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union
from argparse import Namespace
from superflue.config import LOG_LEVEL


def get_log_level(
    args: Optional[Namespace] = None, default_level: int = LOG_LEVEL
) -> int:
    """Get logging level from args or default.

    Args:
        args: Argument namespace that may contain numeric_log_level
        default_level: Default logging level to use if args doesn't specify one

    Returns:
        Logging level as an integer
    """
    if args is not None and hasattr(args, "numeric_log_level"):
        return args.numeric_log_level
    return default_level


def setup_logger(
    name: str,
    log_file: Union[str, Path],
    level: Optional[int] = None,
    args: Optional[Namespace] = None,
) -> logging.Logger:
    """Set up a logger with both file and console handlers.

    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Explicit logging level (deprecated, use args instead)
        args: Arguments namespace that may contain numeric_log_level

    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # Only configure the logger if it hasn't been configured before
    if not logger.handlers:
        # Determine logging level (args takes precedence over level parameter)
        log_level = get_log_level(args, level or LOG_LEVEL)

        # Set logging level
        logger.setLevel(log_level)

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
            backupCount=5,
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
