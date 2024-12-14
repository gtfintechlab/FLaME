from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Mapping
import re
from superflue import RESULTS_DIR, EVALUATION_DIR


def create_standardized_filename(
    task: str, model: str, prefix: Optional[str] = None, timestamp: Optional[str] = None
) -> str:
    """Creates standardized filename with optional prefix and timestamp.

    Args:
        task: Task name (e.g., 'fomc', 'banking77')
        model: Full model name (e.g., 'meta-llama/Llama-2-70b')
        prefix: Optional prefix for the filename (e.g., 'inference', 'evaluation')
        timestamp: Optional timestamp (if not provided, current time will be used)

    Returns:
        Standardized filename in format: {prefix}_{task}_{model}_{timestamp}.csv
        where model has '/' replaced with '_' for filesystem safety
    """
    if not model:
        raise ValueError("Model name cannot be None or empty")

    # Clean model name for filesystem safety
    clean_model = model.replace("/", "_")

    # Ensure we have a timestamp
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build filename parts
    parts = [part for part in [prefix, task, clean_model, timestamp] if part]
    return "_".join(parts) + ".csv"


def get_inference_path(task: str, model: str, timestamp: Optional[str] = None) -> Path:
    """Generate standardized inference results path."""
    if not model:
        raise ValueError("Model name cannot be None or empty")
    if not task:
        raise ValueError("Task name cannot be None or empty")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = create_standardized_filename(task, model, "inference", timestamp)
    path = RESULTS_DIR / task / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_evaluation_path(
    task: str,
    inference_model: str,
    extraction_model: str,
    timestamp: Optional[str] = None,
) -> Path:
    """Generate standardized evaluation results path.

    The filename will include only the inference model name to keep it concise.
    The extraction model details are stored in the metadata.
    """
    if not inference_model:
        raise ValueError("Inference model name cannot be None or empty")
    if not extraction_model:
        raise ValueError("Extraction model name cannot be None or empty")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    clean_model = inference_model.replace("/", "_")
    filename = create_standardized_filename(task, clean_model, "evaluation", timestamp)
    path = EVALUATION_DIR / task / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def extract_model_from_inference_path(path: Path) -> Optional[str]:
    """Extract the model name from an inference results path.

    Example:
        path: /results/fomc/inference_fomc_meta-llama_Llama-2-70b_20240101_120000.csv
        returns: meta-llama/Llama-2-70b
    """
    pattern = r"inference_[^_]+_(.+)_\d{8}_\d{6}\.csv$"
    match = re.search(pattern, path.name)
    if match:
        # Convert back from filename-safe format to original model name
        model = match.group(1)
        return model.replace(
            "_", "/", 1
        )  # Replace only first underscore to handle model names with underscores
    return None


def extract_models_from_evaluation_path(path: Path) -> Optional[str]:
    """Extract inference model name from an evaluation results path.

    Example:
        path: /evaluation/fomc/evaluation_fomc_meta-llama_Llama-2-70b_20240101_120000.csv
        returns: meta-llama/Llama-2-70b
    """
    pattern = r"evaluation_[^_]+_(.+)_\d{8}_\d{6}\.csv$"
    match = re.search(pattern, path.name)
    if match:
        model = match.group(1)
        return model.replace("_", "/", 1)
    return None


def find_related_files(path: Path) -> Mapping[str, Optional[Path]]:
    """Find related files for a given results file.

    For inference files, finds:
    - Metadata file
    - Latest evaluation file using this inference result

    For evaluation files, finds:
    - Metadata file
    - Original inference file
    - Metrics file

    Returns a dictionary mapping file type to path.
    """
    related: Dict[str, Optional[Path]] = {"metadata": path.with_suffix(".meta.json")}

    if "inference_" in path.name:
        # For inference files, find latest evaluation using this result
        inference_model = extract_model_from_inference_path(path)
        if inference_model:
            eval_pattern = f"evaluation_{path.parent.name}_{inference_model.replace('/', '_')}_*.csv"
            eval_files = list(path.parent.parent.glob(eval_pattern))
            related["latest_evaluation"] = max(eval_files) if eval_files else None

    elif "evaluation_" in path.name:
        # For evaluation files, find original inference file and metrics
        inference_model = extract_models_from_evaluation_path(path)
        if inference_model:
            infer_pattern = f"inference_{path.parent.name}_{inference_model.replace('/', '_')}_*.csv"
            infer_files = list(RESULTS_DIR.glob(infer_pattern))
            related["inference"] = max(infer_files) if infer_files else None
            related["metrics"] = path.with_name(f"{path.stem}_metrics.csv")

    return related
