from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Mapping
from superflue.config import RESULTS_DIR, EVALUATION_DIR
import re

def create_standardized_filename(
    task: str,
    model: str,
    prefix: Optional[str] = None,
    timestamp: Optional[str] = None
) -> str:
    """Creates standardized filename with optional prefix and timestamp."""
    clean_model = model.replace('/', '_')  # Handle model names like 'meta-llama/Llama-2'
    timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    parts = [part for part in [prefix, task, clean_model, timestamp] if part]
    return '_'.join(parts) + '.csv'

def get_inference_path(
    task: str,
    model: str,
    timestamp: Optional[str] = None
) -> Path:
    """Generate standardized inference results path."""
    filename = create_standardized_filename(task, model, 'inference', timestamp)
    path = RESULTS_DIR / task / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_evaluation_path(
    task: str,
    inference_model: str,
    extraction_model: str,
    timestamp: Optional[str] = None
) -> Path:
    """Generate standardized evaluation results path.
    
    The filename will include only the inference model name to keep it concise.
    The extraction model details are stored in the metadata.
    """
    clean_model = inference_model.replace('/', '_')
    filename = create_standardized_filename(task, clean_model, 'evaluation', timestamp)
    path = EVALUATION_DIR / task / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def extract_model_from_inference_path(path: Path) -> Optional[str]:
    """Extract the model name from an inference results path.
    
    Example:
        path: /results/fomc/inference_fomc_meta-llama_Llama-2-70b_20240101_120000.csv
        returns: meta-llama/Llama-2-70b
    """
    pattern = r'inference_[^_]+_(.+)_\d{8}_\d{6}\.csv$'
    match = re.search(pattern, path.name)
    if match:
        # Convert back from filename-safe format to original model name
        model = match.group(1)
        return model.replace('_', '/', 1)  # Replace only first underscore to handle model names with underscores
    return None

def extract_models_from_evaluation_path(path: Path) -> Optional[str]:
    """Extract inference model name from an evaluation results path.
    
    Example:
        path: /evaluation/fomc/evaluation_fomc_meta-llama_Llama-2-70b_20240101_120000.csv
        returns: meta-llama/Llama-2-70b
    """
    pattern = r'evaluation_[^_]+_(.+)_\d{8}_\d{6}\.csv$'
    match = re.search(pattern, path.name)
    if match:
        model = match.group(1)
        return model.replace('_', '/', 1)
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
    related: Dict[str, Optional[Path]] = {'metadata': path.with_suffix('.meta.json')}
    
    if 'inference_' in path.name:
        # For inference files, find latest evaluation using this result
        inference_model = extract_model_from_inference_path(path)
        if inference_model:
            eval_pattern = f"evaluation_{path.parent.name}_{inference_model.replace('/', '_')}_*.csv"
            eval_files = list(path.parent.parent.glob(eval_pattern))
            related['latest_evaluation'] = max(eval_files) if eval_files else None
    
    elif 'evaluation_' in path.name:
        # For evaluation files, find original inference file and metrics
        inference_model = extract_models_from_evaluation_path(path)
        if inference_model:
            infer_pattern = f"inference_{path.parent.name}_{inference_model.replace('/', '_')}_*.csv"
            infer_files = list(RESULTS_DIR.glob(infer_pattern))
            related['inference'] = max(infer_files) if infer_files else None
            related['metrics'] = path.with_name(f"{path.stem}_metrics.csv")
    
    return related 