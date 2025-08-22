"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    colored: bool = True,
):
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console: Whether to log to console
        colored: Whether to use colored console output
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if colored and sys.stdout.isatty():
            formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )

        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set level for bench_forge loggers
    logging.getLogger("bench_forge").setLevel(getattr(logging, level.upper()))

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if hasattr(logging, "getLogger"):
        try:
            import litellm  # noqa: F401

            logging.getLogger("litellm").setLevel(logging.WARNING)
        except ImportError:
            pass


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
