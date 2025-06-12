import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flame.config import LOG_CONFIG, LOG_DIR, LOG_FORMAT


def setup_logger(name, log_file, level=None):
    """Set up a logger with both file and console handlers based on global configuration.

    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Optional logging level override. If None, uses component-specific
               level from config or falls back to global level.

    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # Only configure the logger if it hasn't been configured before
    if not logger.handlers:
        # Determine the appropriate logging level
        if level is not None:
            # Use explicitly provided level if given
            logger_level = level
        elif name in LOG_CONFIG["components"]:
            # Look up component-specific level
            logger_level = LOG_CONFIG["components"][name]
        else:
            # Fall back to global level
            logger_level = LOG_CONFIG["level"]

        # Set logging level for this logger
        logger.setLevel(logger_level)

        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT)

        # Add file handler if enabled
        if LOG_CONFIG["file"]["enabled"]:
            # Ensure log directory exists
            log_dir = Path(log_file).parent
            os.makedirs(log_dir, exist_ok=True)

            # Set up file handler with configured parameters
            max_bytes = LOG_CONFIG["file"]["max_size_mb"] * 1024 * 1024
            backup_count = LOG_CONFIG["file"]["backup_count"]
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setLevel(LOG_CONFIG["file"]["level"])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Add console handler if enabled
        if LOG_CONFIG["console"]["enabled"]:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(LOG_CONFIG["console"]["level"])
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False

    return logger


def get_component_logger(component_name, subcomponent=None):
    """Get a logger for a specific component with the appropriate level from config.

    Args:
        component_name: The primary component name (e.g., 'inference', 'evaluation')
        subcomponent: Optional subcomponent name (e.g., 'fomc', 'fpb')

    Returns:
        Configured logger instance
    """
    if subcomponent:
        full_name = f"{component_name}.{subcomponent}"
        log_file = f"{component_name}_{subcomponent}.log"
    else:
        full_name = component_name
        log_file = f"{component_name}.log"

    # Determine log level from component-specific settings
    if full_name in LOG_CONFIG["components"]:
        level = LOG_CONFIG["components"][full_name]
    elif component_name in LOG_CONFIG["components"]:
        level = LOG_CONFIG["components"][component_name]
    else:
        level = LOG_CONFIG["level"]

    return setup_logger(full_name, LOG_DIR / log_file, level)
