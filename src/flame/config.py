"""Configuration settings for the FLaME project.

TEST OUTPUT PATTERN:
When IN_PYTEST is True (set by conftest.py), all outputs should use:
- TEST_OUTPUT_DIR instead of RESULTS_DIR
- TEST_OUTPUT_DIR instead of EVALUATION_DIR

This ensures test artifacts are isolated and gitignored.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

# Base directories
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "src" / "flame"
DATA_DIR = ROOT_DIR / "data"

# Output directories
RESULTS_DIR = ROOT_DIR / "results"
EVALUATION_DIR = ROOT_DIR / "evaluations"  # Changed to match existing directory name
LOG_DIR = ROOT_DIR / "logs"
TEST_OUTPUT_DIR = (
    ROOT_DIR / "tests" / "test_outputs"
)  # Dedicated directory for test artifacts

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RESULTS_DIR,
    EVALUATION_DIR,
    LOG_DIR,
    TEST_OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Default logging settings (can be overridden from YAML config)
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Logging configuration that will be populated from YAML
LOG_CONFIG = {
    "level": DEFAULT_LOG_LEVEL,
    "console": {"enabled": True, "level": DEFAULT_LOG_LEVEL},
    "file": {
        "enabled": True,
        "level": DEFAULT_LOG_LEVEL,
        "max_size_mb": 10,
        "backup_count": 5,
    },
    "components": {
        "litellm": logging.WARNING,
        "inference": DEFAULT_LOG_LEVEL,
        "evaluation": DEFAULT_LOG_LEVEL,
        "batch_utils": DEFAULT_LOG_LEVEL,
    },
}


def configure_logging(config: Optional[Dict] = None):
    """Configure logging settings from a config dictionary.

    Args:
        config: A dictionary containing logging configuration from YAML.
               If None, default settings will be used.
    """
    global LOG_CONFIG

    if not config or "logging" not in config:
        return

    log_config = config["logging"]

    # Parse main logging level
    if "level" in log_config:
        try:
            LOG_CONFIG["level"] = getattr(logging, log_config["level"].upper())
        except AttributeError:
            # Use INFO level as default if invalid level specified
            pass

    # Parse console settings
    if "console" in log_config:
        console_config = log_config["console"]
        if "enabled" in console_config:
            LOG_CONFIG["console"]["enabled"] = console_config["enabled"]
        if "level" in console_config:
            try:
                LOG_CONFIG["console"]["level"] = getattr(
                    logging, console_config["level"].upper()
                )
            except AttributeError:
                # Use default console level if invalid level specified
                pass

    # Parse file settings
    if "file" in log_config:
        file_config = log_config["file"]
        if "enabled" in file_config:
            LOG_CONFIG["file"]["enabled"] = file_config["enabled"]
        if "level" in file_config:
            try:
                LOG_CONFIG["file"]["level"] = getattr(
                    logging, file_config["level"].upper()
                )
            except AttributeError:
                # Use default file level if invalid level specified
                pass
        if "max_size_mb" in file_config:
            LOG_CONFIG["file"]["max_size_mb"] = file_config["max_size_mb"]
        if "backup_count" in file_config:
            LOG_CONFIG["file"]["backup_count"] = file_config["backup_count"]

    # Parse component-specific levels
    if "components" in log_config:
        for component, level_name in log_config["components"].items():
            try:
                level = getattr(logging, level_name.upper())
                # Create the component entry if it doesn't exist
                if component not in LOG_CONFIG["components"]:
                    LOG_CONFIG["components"][component] = level
                else:
                    LOG_CONFIG["components"][component] = level
            except AttributeError:
                # Use default component level if invalid level specified
                pass

    # Ensure litellm is set to WARNING by default if not specified
    if "litellm" not in LOG_CONFIG["components"]:
        LOG_CONFIG["components"]["litellm"] = logging.WARNING

    # Configure litellm if component is present
    configure_litellm()


def configure_litellm():
    """Configure LiteLLM verbosity based on component log level.

    This applies the log level configured for the 'litellm' component
    to control the verbosity of the LiteLLM library.

    Returns:
        The configured litellm logger
    """
    try:
        import logging

        import litellm

        # Get the configured level for litellm
        litellm_level = LOG_CONFIG["components"].get("litellm", logging.WARNING)

        # Always drop params to reduce noise
        litellm.drop_params = True

        # Configure all possible verbosity settings
        if litellm_level <= logging.DEBUG:
            # Debug mode - verbose output (only in debug mode)
            litellm.set_verbose = True
            litellm.verbose = True
            litellm.suppress_debug_info = False
        else:
            # Aggressively suppress all output unless in DEBUG mode
            litellm.set_verbose = False
            litellm.verbose = False
            litellm.suppress_debug_info = True

        # Configure the litellm logger in the Python logging system
        litellm_logger = logging.getLogger("litellm")
        litellm_logger.setLevel(litellm_level)

        # Find and configure all loggers in the litellm hierarchy
        for name in logging.root.manager.loggerDict:
            if (
                name.startswith("litellm")
                or "LiteLLM" in name
                or name in ["httpx", "requests", "openai", "urllib3", "httpcore"]
            ):
                logging.getLogger(name).setLevel(litellm_level)

        # Configure the root logger to WARNING
        logging.getLogger().setLevel(logging.WARNING)

        # Return a properly configured component logger
        from flame.utils.logging_utils import get_component_logger

        return get_component_logger("litellm")

    except ImportError:
        # LiteLLM not available, return a dummy logger
        return logging.getLogger("litellm.dummy")


# For backward compatibility, provide LOG_LEVEL
LOG_LEVEL = DEFAULT_LOG_LEVEL

# Check if running inside pytest
IN_PYTEST = bool(os.environ.get("PYTEST_RUNNING", ""))
