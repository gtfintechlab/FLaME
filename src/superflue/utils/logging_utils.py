import logging
import os
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union
from argparse import Namespace


class IgnoreSpecificWarningFilter(logging.Filter):
    def filter(self, record):
        # Check if the specific unwanted message is in the log record
        if (
            "Only some together models support function calling/response_format."
            in record.getMessage()
        ):
            return False
        return True


def configure_root_logger(
    log_dir: Union[str, Path],
    level: Optional[int] = None,
    args: Optional[Namespace] = None,
) -> None:
    """Configure the root logger with console and file handlers.

    This should be called ONCE at the application entry point.
    """
    root_logger = logging.getLogger()

    # Safety check - if handlers exist and first handler has our formatter,
    # assume logger is already configured
    if root_logger.handlers and hasattr(root_logger.handlers[0], "formatter"):
        if "%(name)s" in root_logger.handlers[0].formatter._fmt:
            return

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Determine logging level
    log_level = level if level is not None else logging.INFO
    root_logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Ensure log directory exists
    try:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Set up file handler with rotation
        file_handler = RotatingFileHandler(
            log_dir / "superflue.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # If file handler fails, log to console only
        root_logger = logging.getLogger()  # Get root logger again to be safe
        root_logger.error(f"Failed to create log file: {e}")
        root_logger.warning("Continuing with console logging only")

    # Set up console handler (always do this, even if file handler fails)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    # Ensure a few silly useless annoying warnings are properly handled
    ignore_filter = IgnoreSpecificWarningFilter()
    root_logger.addFilter(ignore_filter)

    # Set the LiteLLM logger to ERROR to suppress WARNING messages
    lite_llm_logger = logging.getLogger("LiteLLM")
    lite_llm_logger.setLevel(logging.ERROR)

    # Optionally, disable propagation if needed -- this could/should supress all LiteLLM warnings
    # lite_llm_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    This should be used by all modules to get their logger instance.
    The logger will inherit the configuration from the root logger.
    """
    return logging.getLogger(name)


def setup_logger(
    name: str,
    log_file: Union[str, Path],
    level: Optional[int] = None,
    args: Optional[Namespace] = None,
) -> logging.Logger:
    """Deprecated: Use get_logger() instead.

    This function is kept for backward compatibility but will be removed in the future.
    """
    warnings.warn(
        "setup_logger is deprecated. Use get_logger() instead. The reason is that the logger is now configured in the __init__.py file.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_logger(name)
