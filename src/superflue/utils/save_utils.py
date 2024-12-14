from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import json
from datetime import datetime
from .path_utils import get_inference_path, get_evaluation_path


def save_metadata(path: Path, metadata: Dict, timestamp: Optional[str] = None) -> None:
    """Save metadata alongside results."""
    metadata_path = path.with_suffix(".meta.json")
    metadata.update(
        {"timestamp": timestamp or datetime.now().isoformat(), "file_path": str(path)}
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_inference_results(
    df: pd.DataFrame,
    task: str,
    model: str,
    metadata: Optional[Dict] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Save inference results with standardized naming and metadata."""
    path = get_inference_path(task, model, timestamp)
    df.to_csv(path, index=False)
    if metadata:
        save_metadata(path, metadata, timestamp)
    return path


def save_evaluation_results(
    df: pd.DataFrame,
    task: str,
    inference_model: str,
    extraction_model: str,
    metadata: Optional[Dict] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Save evaluation results with standardized naming and metadata."""
    path = get_evaluation_path(task, inference_model, extraction_model, timestamp)
    df.to_csv(path, index=False)
    if metadata:
        save_metadata(path, metadata, timestamp)
    return path
