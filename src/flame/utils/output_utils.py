"""Utilities for generating output filenames and folder hierarchies."""

import uuid
import datetime
import re
from pathlib import Path
from typing import Optional, Tuple

from flame.utils.logging_utils import get_component_logger

logger = get_component_logger("utils.output")


def parse_model_info(model: str) -> Tuple[str, str, Optional[str]]:
    """Parse model string into provider, model family, and model slug.

    Args:
        model: Full model string (e.g., "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct")

    Returns:
        Tuple of (provider, model_slug, model_family) where:
        - provider: e.g., "together_ai"
        - model_slug: e.g., "llama-4-scout-17b-16e-instruct"
        - model_family: e.g., "meta-llama" (optional, can be None)
    """
    parts = model.split("/")

    if len(parts) == 1:
        # Simple model name
        provider = "unknown"
        model_slug = _normalize_model_name(parts[0])
        model_family = None
    elif len(parts) == 2:
        # provider/model
        provider = parts[0]
        model_slug = _normalize_model_name(parts[1])
        model_family = None
    else:
        # provider/family/model
        provider = parts[0]
        model_family = parts[1]
        model_slug = _normalize_model_name(parts[-1])

    return provider, model_slug, model_family


def _normalize_model_name(model_name: str) -> str:
    """Normalize model name to slug format.

    Args:
        model_name: Raw model name

    Returns:
        Normalized slug (lowercase, spaces->hyphens, preserve version tags)
    """
    # Convert to lowercase and replace spaces with hyphens
    normalized = model_name.lower().replace(" ", "-").replace("_", "-")

    # Remove multiple consecutive hyphens
    normalized = re.sub(r"-+", "-", normalized)

    # Remove leading/trailing hyphens
    normalized = normalized.strip("-")

    return normalized


def build_output_directory(
    base_dir: Path, task_slug: str, provider: str, model_family: Optional[str] = None
) -> Path:
    """Build output directory following the new hierarchy scheme.

    Args:
        base_dir: Base output directory (RESULTS_DIR or EVALUATION_DIR)
        task_slug: Task name slug
        provider: Provider name (e.g., "together_ai")
        model_family: Optional model family (e.g., "meta-llama")

    Returns:
        Path object for the output directory
    """
    if model_family:
        output_dir = base_dir / task_slug / provider / model_family
    else:
        output_dir = base_dir / task_slug / provider

    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def build_output_filename(
    model_slug: str, task_slug: str, run: int = 1, metrics: bool = False
) -> str:
    """Build output filename following the new template.

    Template: "{model_slug}__{task_slug}__r{run:02d}__{yyyymmdd}__{uuid8}{suffix}.csv"

    Args:
        model_slug: Model name slug (e.g., "llama-4-scout-17b-16e-instruct")
        task_slug: Task name slug (e.g., "causal_classification")
        run: Run number (default 1)
        metrics: Whether this is a metrics file (adds "_metrics" suffix)

    Returns:
        Filename string
    """
    date = datetime.datetime.now().strftime("%Y%m%d")
    uid8 = uuid.uuid4().hex[:8]
    suffix = "_metrics" if metrics else ""

    filename = f"{model_slug}__{task_slug}__r{run:02d}__{date}__{uid8}{suffix}.csv"
    return filename


def generate_output_path(
    base_dir: Path, task: str, model: str, run: int = 1, metrics: bool = False
) -> Path:
    """Generate complete output path with new hierarchy and filename scheme.

    Args:
        base_dir: Base output directory (RESULTS_DIR or EVALUATION_DIR)
        task: Task name
        model: Full model string
        run: Run number (default 1)
        metrics: Whether this is a metrics file

    Returns:
        Complete Path object for the output file
    """
    # Parse model information
    provider, model_slug, model_family = parse_model_info(model)

    # Build directory
    output_dir = build_output_directory(base_dir, task, provider, model_family)

    # Build filename
    filename = build_output_filename(model_slug, task, run, metrics)

    full_path = output_dir / filename

    logger.debug(f"Generated output path: {full_path}")
    return full_path


def parse_output_filename(filename: str) -> dict:
    """Parse an output filename back into its components.

    Args:
        filename: Filename to parse (with or without .csv extension)

    Returns:
        Dictionary with keys: model_slug, task_slug, run, date, uuid8, metrics
    """
    # Remove .csv extension if present
    name = filename.replace(".csv", "")

    # Check for metrics suffix
    metrics = name.endswith("_metrics")
    if metrics:
        name = name[:-8]  # Remove "_metrics"

    # Split by double underscores
    parts = name.split("__")

    if len(parts) != 5:
        raise ValueError(f"Invalid filename format: {filename}")

    model_slug, task_slug, run_part, date_part, uuid_part = parts

    # Parse run number
    if not run_part.startswith("r") or len(run_part) != 3:
        raise ValueError(f"Invalid run format in filename: {filename}")

    try:
        run = int(run_part[1:])
    except ValueError:
        raise ValueError(f"Invalid run number in filename: {filename}")

    # Validate date and uuid parts
    if len(date_part) != 8:  # YYYYMMDD
        raise ValueError(f"Invalid date format in filename: {filename}")

    if len(uuid_part) != 8:  # 8 hex chars
        raise ValueError(f"Invalid UUID format in filename: {filename}")

    date = date_part
    uuid8 = uuid_part

    return {
        "model_slug": model_slug,
        "task_slug": task_slug,
        "run": run,
        "date": date,
        "uuid8": uuid8,
        "metrics": metrics,
    }


# Legacy compatibility functions
def generate_inference_filename(
    task: str, model: str, output_dir: Optional[Path] = None
) -> Path:
    """Legacy function for backwards compatibility - use generate_output_path instead.

    This function maintains compatibility with existing code while using the new structure.
    """
    if output_dir is None:
        # Import at runtime to get patched values in tests
        from flame.config import RESULTS_DIR, TEST_OUTPUT_DIR, IN_PYTEST

        output_dir = TEST_OUTPUT_DIR if IN_PYTEST else RESULTS_DIR

    return generate_output_path(output_dir, task, model)


def generate_evaluation_filename(
    task: str,
    model: str,
    output_dir: Optional[Path] = None,
    run: int = 1,
    metrics: bool = False,
) -> Path:
    """Generate evaluation filename with new structure.

    Args:
        task: Task name
        model: Full model string
        output_dir: Base evaluation directory (optional)
        run: Run number (default 1)
        metrics: Whether this is a metrics file

    Returns:
        Complete Path object for the evaluation file
    """
    if output_dir is None:
        # Import at runtime to get patched values in tests
        from flame.config import EVALUATION_DIR, TEST_OUTPUT_DIR, IN_PYTEST

        output_dir = TEST_OUTPUT_DIR if IN_PYTEST else EVALUATION_DIR

    return generate_output_path(output_dir, task, model, run, metrics)
