"""Configuration management for BenchForge.

Professional-grade configuration system with multiple sources,
environment variable support, and validation.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading
from functools import lru_cache

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BenchForgeConfig:
    """Main configuration for BenchForge with comprehensive settings."""

    # Project settings
    project_name: str = "benchforge"
    version: str = "0.3.0"
    description: str = "Modern benchmark engine for LLM evaluation"

    # Environment
    environment: str = "development"  # development, testing, production
    debug_mode: bool = False
    strict_mode: bool = False  # Fail on warnings

    # Directories
    base_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    output_dir: str = "outputs"
    log_dir: str = "logs"
    data_dir: str = "data"
    temp_dir: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = True
    log_to_console: bool = True
    log_file_max_bytes: int = 10485760  # 10MB
    log_file_backup_count: int = 5
    colored_logs: bool = True

    # Processing
    default_batch_size: int = 10
    max_parallel_workers: int = 4
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True

    # Caching - DISABLED for research integrity
    enable_cache: bool = False  # Changed from True - we need independent API calls
    cache_ttl: int = 86400  # 24 hours
    cache_max_size_mb: float = 1000
    cache_compression: bool = False
    response_cache_enabled: bool = False  # Changed from True - no response reuse

    # LLM defaults
    default_provider: str = "litellm"
    default_model: str = "gpt-3.5-turbo"
    default_max_tokens: int = 256
    default_temperature: float = 0.0
    default_top_p: float = 1.0
    default_seed: Optional[int] = 42

    # API settings
    api_key_env_var: str = "BENCHFORGE_API_KEY"
    api_base_url: Optional[str] = None
    api_version: Optional[str] = None
    api_org_id: Optional[str] = None

    # Metrics
    default_metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1_macro"])
    metric_decimal_places: int = 4

    # Output settings
    output_format: str = "json"  # json, csv, parquet
    pretty_print: bool = True
    include_timestamps: bool = True
    include_metadata: bool = True

    # Validation
    validate_inputs: bool = True
    validate_outputs: bool = True
    allow_empty_datasets: bool = False
    min_dataset_size: int = 1

    # Performance
    profile_enabled: bool = False
    memory_limit_mb: Optional[float] = None
    cpu_limit_percent: Optional[float] = None

    # Security
    allow_remote_code: bool = False
    sanitize_inputs: bool = True
    mask_sensitive_data: bool = True

    # Experimental features
    enable_experimental: bool = False
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Set up base directory
        if self.base_dir is None:
            self.base_dir = str(Path.cwd())

        # Set up cache directory
        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".cache" / "benchforge")

        # Set up temp directory
        if self.temp_dir is None:
            import tempfile

            self.temp_dir = tempfile.gettempdir()

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(
                f"Invalid log_level: {self.log_level}. Must be one of {valid_levels}"
            )

        # Validate environment
        valid_envs = ["development", "testing", "production"]
        if self.environment not in valid_envs:
            raise ValueError(
                f"Invalid environment: {self.environment}. Must be one of {valid_envs}"
            )

        # Apply environment-specific defaults
        if self.environment == "production":
            self.debug_mode = False
            self.strict_mode = True
            self.validate_inputs = True
            self.validate_outputs = True
            self.allow_remote_code = False
        elif self.environment == "testing":
            self.debug_mode = True
            self.cache_ttl = 60  # Shorter TTL for testing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchForgeConfig":
        """Create from dictionary with validation.

        Args:
            data: Configuration dictionary

        Returns:
            BenchForgeConfig instance
        """
        # Filter out unknown fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Warn about unknown fields
        unknown_fields = set(data.keys()) - valid_fields
        if unknown_fields:
            logger.warning(f"Ignoring unknown config fields: {unknown_fields}")

        return cls(**filtered_data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "BenchForgeConfig":
        """Load from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            BenchForgeConfig instance
        """
        path = Path(path)

        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}

            logger.info(f"Loaded config from: {path}")
            return cls.from_dict(data)

        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            raise

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BenchForgeConfig":
        """Load from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            BenchForgeConfig instance
        """
        path = Path(path)

        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        try:
            with open(path, "r") as f:
                data = json.load(f)

            logger.info(f"Loaded config from: {path}")
            return cls.from_dict(data)

        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            raise

    def save_yaml(self, path: Union[str, Path]):
        """Save to YAML file.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config to: {path}")

    def save_json(self, path: Union[str, Path]):
        """Save to JSON file.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved config to: {path}")

    def update(self, **kwargs) -> "BenchForgeConfig":
        """Update configuration with new values.

        Args:
            **kwargs: Configuration values to update

        Returns:
            Updated config instance
        """
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return BenchForgeConfig.from_dict(current_dict)

    def validate(self) -> List[str]:
        """Validate configuration and return any issues.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check directories
        if self.cache_max_size_mb <= 0:
            issues.append(
                f"cache_max_size_mb must be positive: {self.cache_max_size_mb}"
            )

        if self.default_batch_size < 1:
            issues.append(
                f"default_batch_size must be at least 1: {self.default_batch_size}"
            )

        if self.max_parallel_workers < 1:
            issues.append(
                f"max_parallel_workers must be at least 1: {self.max_parallel_workers}"
            )

        if self.default_temperature < 0 or self.default_temperature > 2:
            issues.append(
                f"default_temperature must be in [0, 2]: {self.default_temperature}"
            )

        if self.default_top_p < 0 or self.default_top_p > 1:
            issues.append(f"default_top_p must be in [0, 1]: {self.default_top_p}")

        return issues


class ConfigManager:
    """Manage BenchForge configuration with singleton pattern."""

    _instance = None
    _lock = threading.Lock()
    _config: Optional[BenchForgeConfig] = None

    def __new__(cls):
        """Ensure single instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize config manager."""
        if self._config is None:
            self._config = self._load_config()

    @lru_cache(maxsize=1)
    def _load_config(self) -> BenchForgeConfig:
        """Load configuration from various sources with caching.

        Priority order:
        1. Environment variables (BENCHFORGE_*)
        2. Local config file (./benchforge.yaml or ./.benchforge.yaml)
        3. User config file (~/.benchforge/config.yaml)
        4. System config file (/etc/benchforge/config.yaml)
        5. Default configuration

        Returns:
            Merged configuration
        """
        # Start with defaults
        config = BenchForgeConfig()

        # System config (lowest priority)
        system_config = Path("/etc/benchforge/config.yaml")
        if system_config.exists():
            try:
                logger.info(f"Loading system config: {system_config}")
                system_cfg = BenchForgeConfig.from_yaml(system_config)
                config = self._merge_configs(config, system_cfg)
            except Exception as e:
                logger.warning(f"Failed to load system config: {e}")

        # User config
        user_config = Path.home() / ".benchforge" / "config.yaml"
        if user_config.exists():
            try:
                logger.info(f"Loading user config: {user_config}")
                user_cfg = BenchForgeConfig.from_yaml(user_config)
                config = self._merge_configs(config, user_cfg)
            except Exception as e:
                logger.warning(f"Failed to load user config: {e}")

        # Local config (highest file priority)
        local_configs = [
            "benchforge.yaml",
            "benchforge.yml",
            ".benchforge.yaml",
            ".benchforge.yml",
            "benchforge.json",
            ".benchforge.json",
        ]

        for config_name in local_configs:
            local_config = Path(config_name)
            if local_config.exists():
                try:
                    logger.info(f"Loading local config: {local_config}")
                    if config_name.endswith((".json")):
                        local_cfg = BenchForgeConfig.from_json(local_config)
                    else:
                        local_cfg = BenchForgeConfig.from_yaml(local_config)
                    config = self._merge_configs(config, local_cfg)
                    break  # Use first found
                except Exception as e:
                    logger.warning(f"Failed to load local config: {e}")

        # Override with environment variables (highest priority)
        config = self._apply_env_overrides(config)

        # Validate final configuration
        issues = config.validate()
        if issues:
            if config.strict_mode:
                raise ValueError(f"Configuration validation failed: {issues}")
            else:
                for issue in issues:
                    logger.warning(f"Config validation issue: {issue}")

        return config

    def _merge_configs(
        self, base: BenchForgeConfig, override: BenchForgeConfig
    ) -> BenchForgeConfig:
        """Merge two configurations with intelligent handling.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        base_dict = base.to_dict()
        override_dict = override.to_dict()

        # Merge with special handling for lists and dicts
        for key, value in override_dict.items():
            if value is not None:
                if isinstance(value, list) and key in base_dict:
                    # For lists, replace entirely (don't append)
                    base_dict[key] = value
                elif isinstance(value, dict) and key in base_dict:
                    # For dicts, merge recursively
                    if isinstance(base_dict[key], dict):
                        base_dict[key] = {**base_dict[key], **value}
                    else:
                        base_dict[key] = value
                else:
                    base_dict[key] = value

        return BenchForgeConfig.from_dict(base_dict)

    def _apply_env_overrides(self, config: BenchForgeConfig) -> BenchForgeConfig:
        """Apply environment variable overrides with type conversion.

        Environment variables format: BENCHFORGE_<SETTING>
        Example: BENCHFORGE_LOG_LEVEL=DEBUG

        Args:
            config: Configuration to override

        Returns:
            Updated configuration
        """
        config_dict = config.to_dict()
        prefix = "BENCHFORGE_"

        for key in config_dict.keys():
            env_key = f"{prefix}{key.upper()}"
            env_value = os.environ.get(env_key)

            if env_value is not None:
                # Convert string to appropriate type
                current_value = config_dict[key]

                try:
                    if isinstance(current_value, bool):
                        config_dict[key] = env_value.lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                        )
                    elif isinstance(current_value, int):
                        config_dict[key] = int(env_value)
                    elif isinstance(current_value, float):
                        config_dict[key] = float(env_value)
                    elif isinstance(current_value, list):
                        # Parse comma-separated list
                        config_dict[key] = [v.strip() for v in env_value.split(",")]
                    elif isinstance(current_value, dict):
                        # Parse JSON string
                        config_dict[key] = json.loads(env_value)
                    elif current_value is None or isinstance(current_value, str):
                        config_dict[key] = env_value
                    else:
                        logger.warning(
                            f"Cannot convert env var {env_key} to type {type(current_value)}"
                        )
                        continue

                    logger.debug(f"Override from env: {key} = {config_dict[key]}")

                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to parse env var {env_key}: {e}")

        return BenchForgeConfig.from_dict(config_dict)

    def get(self) -> BenchForgeConfig:
        """Get current configuration.

        Returns:
            Current configuration
        """
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def set(self, config: BenchForgeConfig):
        """Set configuration.

        Args:
            config: New configuration
        """
        # Validate before setting
        issues = config.validate()
        if issues and config.strict_mode:
            raise ValueError(f"Invalid configuration: {issues}")

        self._config = config
        self._load_config.cache_clear()  # Clear cache
        logger.info("Configuration updated")

    def reload(self):
        """Reload configuration from sources."""
        self._load_config.cache_clear()  # Clear cache
        self._config = self._load_config()
        logger.info("Configuration reloaded")

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key with dot notation support.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if not found

        Returns:
            Configuration value
        """
        config = self.get()
        config_dict = config.to_dict()

        # Support dot notation (e.g., "feature_flags.new_feature")
        keys = key.split(".")
        value = config_dict

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set_value(self, key: str, value: Any):
        """Set configuration value by key.

        Args:
            key: Configuration key
            value: New value
        """
        config = self.get()
        config_dict = config.to_dict()

        # Support dot notation
        keys = key.split(".")
        target = config_dict

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

        # Create new config
        new_config = BenchForgeConfig.from_dict(config_dict)
        self.set(new_config)


# Global config manager instance
_manager = ConfigManager()


def get_config() -> BenchForgeConfig:
    """Get global configuration.

    Returns:
        Global configuration
    """
    return _manager.get()


def set_config(config: BenchForgeConfig):
    """Set global configuration.

    Args:
        config: New configuration
    """
    _manager.set(config)


def reload_config():
    """Reload configuration from sources."""
    _manager.reload()


def get_config_value(key: str, default: Any = None) -> Any:
    """Get specific configuration value.

    Args:
        key: Configuration key (supports dot notation)
        default: Default value

    Returns:
        Configuration value
    """
    return _manager.get_value(key, default)


def set_config_value(key: str, value: Any):
    """Set specific configuration value.

    Args:
        key: Configuration key
        value: New value
    """
    _manager.set_value(key, value)


# Module exports
__all__ = [
    "BenchForgeConfig",
    "ConfigManager",
    "get_config",
    "set_config",
    "reload_config",
    "get_config_value",
    "set_config_value",
]
