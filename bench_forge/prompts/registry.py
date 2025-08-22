"""Prompt registry with versioning and management."""

from typing import Any, Callable, Dict, List, Optional, Union
import logging
from threading import Lock
from datetime import datetime
import hashlib

from bench_forge.prompts.formats import PromptFormat, PromptComponents
from bench_forge.prompts.templates import PromptTemplate, create_template


logger = logging.getLogger(__name__)


class PromptVersion:
    """Container for versioned prompt."""

    def __init__(
        self,
        function: Callable,
        version: str,
        format_type: PromptFormat,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize prompt version.

        Args:
            function: Prompt function
            version: Version string
            format_type: Prompt format
            metadata: Optional metadata
        """
        self.function = function
        self.version = version
        self.format_type = format_type
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.usage_count = 0

        # Generate hash for this version
        content = f"{function.__name__}:{version}:{format_type.value}"
        self.hash = hashlib.md5(content.encode()).hexdigest()[:8]

    def __call__(self, *args, **kwargs):
        """Call the prompt function."""
        self.usage_count += 1
        return self.function(*args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "format": self.format_type.value,
            "created_at": self.created_at.isoformat(),
            "usage_count": self.usage_count,
            "hash": self.hash,
            "metadata": self.metadata,
        }


class PromptRegistry:
    """Registry for prompt templates with versioning support."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Ensure single instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry (only once)."""
        if self._initialized:
            return

        # Structure: {task: {format: {version: PromptVersion}}}
        self._prompts: Dict[str, Dict[str, Dict[str, PromptVersion]]] = {}

        # Template cache
        self._template_cache: Dict[str, PromptTemplate] = {}

        # Statistics
        self._stats = {"total_prompts": 0, "total_calls": 0, "cache_hits": 0}

        self._initialized = True
        logger.info("PromptRegistry initialized")

    def register(
        self,
        task: str,
        format_type: Union[str, PromptFormat],
        version: str = "1.0.0",
        function: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register a prompt function.

        Args:
            task: Task name
            format_type: Prompt format
            version: Version string
            function: Prompt function (if not used as decorator)
            metadata: Optional metadata

        Returns:
            Decorator function or None
        """
        if isinstance(format_type, str):
            format_type = PromptFormat(format_type)

        def decorator(func: Callable) -> Callable:
            # Initialize nested dictionaries
            if task not in self._prompts:
                self._prompts[task] = {}

            format_key = format_type.value
            if format_key not in self._prompts[task]:
                self._prompts[task][format_key] = {}

            # Check for duplicate version
            if version in self._prompts[task][format_key]:
                logger.warning(f"Overwriting prompt: {task}/{format_key}/{version}")

            # Create and store prompt version
            prompt_version = PromptVersion(func, version, format_type, metadata)
            self._prompts[task][format_key][version] = prompt_version

            self._stats["total_prompts"] += 1
            logger.info(f"Registered prompt: {task}/{format_key}/{version}")

            return func

        if function is not None:
            return decorator(function)

        return decorator

    def get(
        self,
        task: str,
        format_type: Union[str, PromptFormat, None] = None,
        version: str = "latest",
    ) -> Callable:
        """Get a prompt function.

        Args:
            task: Task name
            format_type: Prompt format (optional)
            version: Version string or "latest"

        Returns:
            Prompt function

        Raises:
            KeyError: If prompt not found
        """
        if task not in self._prompts:
            raise KeyError(f"Task '{task}' not found in registry")

        # If format not specified, try to find any format
        if format_type is None:
            if not self._prompts[task]:
                raise KeyError(f"No prompts registered for task '{task}'")
            # Use first available format
            format_key = next(iter(self._prompts[task].keys()))
        else:
            if isinstance(format_type, PromptFormat):
                format_key = format_type.value
            else:
                format_key = format_type

        if format_key not in self._prompts[task]:
            available = list(self._prompts[task].keys())
            raise KeyError(
                f"Format '{format_key}' not found for task '{task}'. Available: {available}"
            )

        versions = self._prompts[task][format_key]

        if not versions:
            raise KeyError(f"No versions found for {task}/{format_key}")

        # Handle version selection
        if version == "latest":
            # Get the most recent version (by creation time)
            prompt_version = max(versions.values(), key=lambda v: v.created_at)
        elif version in versions:
            prompt_version = versions[version]
        else:
            available = list(versions.keys())
            raise KeyError(f"Version '{version}' not found. Available: {available}")

        self._stats["total_calls"] += 1
        return prompt_version

    def list_tasks(self) -> List[str]:
        """List all registered tasks.

        Returns:
            List of task names
        """
        return list(self._prompts.keys())

    def list_formats(self, task: str) -> List[str]:
        """List formats for a task.

        Args:
            task: Task name

        Returns:
            List of format names
        """
        if task not in self._prompts:
            return []
        return list(self._prompts[task].keys())

    def list_versions(
        self, task: str, format_type: Union[str, PromptFormat]
    ) -> List[str]:
        """List versions for a task/format combination.

        Args:
            task: Task name
            format_type: Prompt format

        Returns:
            List of version strings
        """
        if isinstance(format_type, PromptFormat):
            format_key = format_type.value
        else:
            format_key = format_type

        if task not in self._prompts or format_key not in self._prompts[task]:
            return []

        return list(self._prompts[task][format_key].keys())

    def get_info(
        self,
        task: str,
        format_type: Optional[Union[str, PromptFormat]] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get information about registered prompts.

        Args:
            task: Task name
            format_type: Optional format filter
            version: Optional version filter

        Returns:
            Information dictionary
        """
        if task not in self._prompts:
            return {}

        info = {"task": task, "formats": {}}

        for format_key, versions in self._prompts[task].items():
            if format_type and format_key != str(format_type):
                continue

            format_info = {"versions": {}}

            for ver, prompt_ver in versions.items():
                if version and ver != version:
                    continue

                format_info["versions"][ver] = prompt_ver.to_dict()

            if format_info["versions"]:
                info["formats"][format_key] = format_info

        return info

    def create_template(
        self,
        task: str,
        format_type: Union[str, PromptFormat],
        components: Optional[PromptComponents] = None,
        version: str = "latest",
        **kwargs,
    ) -> PromptTemplate:
        """Create a template instance for a registered prompt.

        Args:
            task: Task name
            format_type: Prompt format
            components: Prompt components
            version: Version to use
            **kwargs: Additional template arguments

        Returns:
            PromptTemplate instance
        """
        # Check cache first
        cache_key = f"{task}:{format_type}:{version}"

        if cache_key in self._template_cache:
            self._stats["cache_hits"] += 1
            template = self._template_cache[cache_key]
            # Update components if provided
            if components:
                template.components = components
            return template

        # Get prompt function
        self.get(task, format_type, version)

        # Create template
        if isinstance(format_type, str):
            format_type = PromptFormat(format_type)

        template = create_template(format_type, components, **kwargs)

        # Cache template
        self._template_cache[cache_key] = template

        return template

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Statistics dictionary
        """
        stats = self._stats.copy()
        stats["num_tasks"] = len(self._prompts)
        stats["cache_size"] = len(self._template_cache)

        # Count total versions
        total_versions = 0
        for task_formats in self._prompts.values():
            for versions in task_formats.values():
                total_versions += len(versions)
        stats["total_versions"] = total_versions

        # Cache hit rate
        if stats["total_calls"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_calls"]
        else:
            stats["cache_hit_rate"] = 0.0

        return stats

    def clear_cache(self):
        """Clear template cache."""
        self._template_cache.clear()
        logger.info("Cleared template cache")


# Global registry instance
_registry = PromptRegistry()


def get_prompt_registry() -> PromptRegistry:
    """Get the global prompt registry.

    Returns:
        PromptRegistry instance
    """
    return _registry


def register_prompt(
    task: str,
    format_type: Union[str, PromptFormat],
    template: str,
    examples: Optional[List[Dict[str, str]]] = None,
    variables: Optional[List[str]] = None,
    version: str = "1.0.0",
    **metadata,
) -> None:
    """Register a prompt template with the global registry.

    Args:
        task: Task name
        format_type: Prompt format type
        template: Prompt template string
        examples: Optional examples for few-shot prompts
        variables: Template variables
        version: Version string
        **metadata: Additional metadata
    """
    registry = get_prompt_registry()
    registry.register(
        task=task,
        format_type=format_type,
        template=template,
        examples=examples,
        variables=variables,
        version=version,
        **metadata,
    )


def prompt(
    task: str,
    format_type: Union[str, PromptFormat] = PromptFormat.ZERO_SHOT,
    version: str = "1.0.0",
    **metadata,
):
    """Decorator to register a prompt.

    Args:
        task: Task name
        format_type: Prompt format
        version: Version string
        **metadata: Additional metadata

    Returns:
        Decorator function
    """
    registry = get_prompt_registry()
    return registry.register(task, format_type, version, metadata=metadata)
