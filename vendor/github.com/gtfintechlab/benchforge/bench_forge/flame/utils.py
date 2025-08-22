"""Utility functions for FLAME integration with BenchForge.

This module provides helper functions and utilities specifically
designed for FLAME workflows.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import pandas as pd

from bench_forge.llm.config import LLMConfig
from bench_forge.flame.adapter import FLAMEConfig
from bench_forge.tasks.config import PromptFormat
from bench_forge.data.loader import loader_factory

logger = logging.getLogger(__name__)


def args_to_config(args, task_name: Optional[str] = None) -> Dict[str, Any]:
    """Convert command-line arguments to configuration dictionaries.

    This function converts FLAME command-line arguments into
    both LLMConfig and FLAMEConfig dictionaries.

    Args:
        args: Command-line arguments (e.g., from argparse)
        task_name: Optional task name override

    Returns:
        Dictionary with 'llm_config' and 'task_config' keys
    """
    # Determine task name
    if task_name is None:
        task_name = getattr(args, "task", None) or getattr(args, "dataset", "unknown")

    # Create LLM configuration
    llm_config = LLMConfig(
        provider="litellm",  # FLAME uses LiteLLM
        model=getattr(
            args, "model", "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
        ),
        max_tokens=getattr(args, "max_tokens", 256),
        temperature=getattr(args, "temperature", 0.0),
        top_p=getattr(args, "top_p", 1.0),
        top_k=getattr(args, "top_k", None),
        seed=getattr(args, "seed", 42),
        timeout=getattr(args, "timeout", 60),
        max_retries=getattr(args, "max_retries", 3),
        cache_responses=True,
    )

    # Determine prompt format
    prompt_format = PromptFormat.ZERO_SHOT
    if hasattr(args, "prompt_format"):
        format_str = args.prompt_format.lower()
        if "few" in format_str:
            prompt_format = PromptFormat.FEW_SHOT
        elif "chain" in format_str or "cot" in format_str:
            prompt_format = PromptFormat.CHAIN_OF_THOUGHT

    # Map task names to HuggingFace datasets
    huggingface_datasets = {
        "fomc": "gtfintechlab/fomc_communication",
        "fpb": "financial_phrasebank",
        # Add more mappings as needed
    }

    # Create task-specific configuration
    # Import task-specific configs if available
    if task_name == "fomc":
        from bench_forge.flame.tasks.fomc import FOMCConfig

        task_config = FOMCConfig(
            name=task_name,
            model=getattr(args, "model", None),
            prompt_format=prompt_format,
            batch_size=getattr(args, "batch_size", 10),
            max_tokens=getattr(args, "max_tokens", 256),
            num_samples=getattr(args, "num_samples", None),
        )
        # FOMCConfig already has the correct text_field="sentence"
    else:
        # Generic config for other tasks
        task_config = FLAMEConfig(
            name=task_name,
            dataset=getattr(args, "dataset", task_name),
            huggingface_dataset=huggingface_datasets.get(
                task_name
            ),  # Add HuggingFace dataset
            dataset_split=getattr(args, "split", "test"),
            metrics=getattr(args, "metrics", ["accuracy", "f1_macro"]),
            prompt_format=prompt_format,
            batch_size=getattr(args, "batch_size", 10),
            max_tokens=getattr(args, "max_tokens", 256),
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            top_k=getattr(args, "top_k", None),
            seed=getattr(args, "seed", 42),
            num_samples=getattr(args, "num_samples", None),
            # FLAME-specific
            results_dir=Path(getattr(args, "results_dir", None) or "results")
            / task_name,
            evaluation_dir=Path(getattr(args, "evaluation_dir", None) or "evaluations")
            / task_name,
        )

    return {
        "llm_config": llm_config,
        "task_config": task_config,
    }


def load_flame_dataset(
    dataset_name: str,
    split: str = "test",
    cache_dir: Optional[Path] = None,
    trust_remote_code: bool = True,
    **kwargs,
) -> Union[List[Dict], pd.DataFrame]:
    """Load a FLAME dataset with proper handling.

    Args:
        dataset_name: Name or path of the dataset
        split: Dataset split to load
        cache_dir: Cache directory for datasets
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for loader

    Returns:
        Loaded dataset as list of dicts or DataFrame
    """
    cache_dir = cache_dir or Path(".cache") / "flame_datasets"

    # Try HuggingFace first
    if "/" in dataset_name or dataset_name.startswith("gtfintechlab"):
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")
        try:
            from datasets import load_dataset

            dataset = load_dataset(
                dataset_name,
                split=split,
                trust_remote_code=trust_remote_code,
                cache_dir=str(cache_dir),
                **kwargs,
            )

            # Convert to list of dicts
            if hasattr(dataset, "to_dict"):
                return dataset.to_dict()["data"]
            elif hasattr(dataset, "__iter__"):
                return list(dataset)
            else:
                return dataset

        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace: {e}")

    # Try local file
    path = Path(dataset_name)
    if path.exists():
        logger.info(f"Loading local dataset: {path}")
        loader = loader_factory(str(path))
        return loader.load(str(path))

    # Try standard FLAME datasets location
    flame_data_dir = Path("data") / dataset_name
    if flame_data_dir.exists():
        # Look for data files
        for pattern in ["*.csv", "*.json", "*.parquet"]:
            files = list(flame_data_dir.glob(pattern))
            if files:
                logger.info(f"Loading FLAME dataset from: {files[0]}")
                loader = loader_factory(str(files[0]))
                return loader.load(str(files[0]))

    raise ValueError(f"Could not load dataset: {dataset_name}")


def process_flame_results(
    results_df: pd.DataFrame,
    task_name: str,
    model_name: str,
    output_dir: Optional[Path] = None,
    include_metadata: bool = True,
) -> Path:
    """Process and save FLAME results in standard format.

    Args:
        results_df: Results DataFrame
        task_name: Name of the task
        model_name: Name of the model
        output_dir: Output directory for results
        include_metadata: Whether to include metadata

    Returns:
        Path to saved results file
    """
    output_dir = output_dir or Path("results") / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean model name for filename
    model_name_clean = model_name.replace("/", "_").replace(":", "_")

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%d_%m_%Y_%H%M")
    filename = f"{task_name}_{model_name_clean}_{timestamp}.csv"
    output_path = output_dir / filename

    # Add metadata if requested
    if include_metadata:
        results_df.attrs["task"] = task_name
        results_df.attrs["model"] = model_name
        results_df.attrs["timestamp"] = timestamp
        results_df.attrs["num_samples"] = len(results_df)

        # Add summary statistics
        if (
            "extracted_response" in results_df.columns
            and "ground_truth" in results_df.columns
        ):
            correct = (
                results_df["extracted_response"] == results_df["ground_truth"]
            ).sum()
            results_df.attrs["accuracy"] = correct / len(results_df)

    # Save results
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved results to: {output_path}")

    # Also save metadata separately
    if include_metadata:
        metadata_path = output_path.with_suffix(".meta.json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(results_df.attrs, f, indent=2, default=str)

    return output_path


def validate_flame_results(results_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate FLAME results file format and content.

    Args:
        results_path: Path to results file

    Returns:
        Validation report dictionary
    """
    results_path = Path(results_path)

    if not results_path.exists():
        return {"valid": False, "error": f"Results file not found: {results_path}"}

    try:
        # Load results
        df = pd.read_csv(results_path)

        # Check required columns
        required_columns = ["input", "prompt", "raw_response", "extracted_response"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return {
                "valid": False,
                "error": f"Missing required columns: {missing_columns}",
                "columns": list(df.columns),
            }

        # Validate content
        report = {
            "valid": True,
            "num_samples": len(df),
            "columns": list(df.columns),
            "null_counts": df[required_columns].isnull().sum().to_dict(),
        }

        # Check for ground truth
        if "ground_truth" in df.columns:
            report["has_ground_truth"] = True
            report["unique_labels"] = df["ground_truth"].nunique()
        else:
            report["has_ground_truth"] = False

        # Check extraction success
        if "extracted_response" in df.columns:
            extraction_failures = df["extracted_response"].isnull().sum()
            report["extraction_failures"] = int(extraction_failures)
            report["extraction_success_rate"] = 1 - (extraction_failures / len(df))

        return report

    except Exception as e:
        return {"valid": False, "error": f"Failed to validate results: {str(e)}"}


def merge_flame_results(
    results_files: List[Union[str, Path]],
    output_path: Optional[Path] = None,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Merge multiple FLAME results files.

    Args:
        results_files: List of results file paths
        output_path: Optional path to save merged results
        deduplicate: Whether to remove duplicate entries

    Returns:
        Merged results DataFrame
    """
    dfs = []

    for file_path in results_files:
        try:
            df = pd.read_csv(file_path)
            # Add source file info
            df["source_file"] = str(Path(file_path).name)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} results from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")

    if not dfs:
        raise ValueError("No results files could be loaded")

    # Merge DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates if requested
    if deduplicate:
        before_count = len(merged_df)
        # Deduplicate based on input and model response
        merged_df = merged_df.drop_duplicates(
            subset=["input", "raw_response"], keep="first"
        )
        after_count = len(merged_df)

        if before_count > after_count:
            logger.info(f"Removed {before_count - after_count} duplicate entries")

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Saved merged results to: {output_path}")

    return merged_df


def compute_task_statistics(
    results_df: pd.DataFrame, task_name: Optional[str] = None
) -> Dict[str, Any]:
    """Compute comprehensive statistics for FLAME task results.

    Args:
        results_df: Results DataFrame
        task_name: Optional task name for context

    Returns:
        Dictionary of statistics
    """
    stats = {
        "task": task_name or "unknown",
        "num_samples": len(results_df),
    }

    # Response statistics
    if "raw_response" in results_df.columns:
        response_lengths = results_df["raw_response"].str.len()
        stats["response_stats"] = {
            "mean_length": float(response_lengths.mean()),
            "std_length": float(response_lengths.std()),
            "min_length": int(response_lengths.min()),
            "max_length": int(response_lengths.max()),
        }

    # Extraction statistics
    if "extracted_response" in results_df.columns:
        extraction_failures = results_df["extracted_response"].isnull().sum()
        stats["extraction_stats"] = {
            "failures": int(extraction_failures),
            "success_rate": float(1 - (extraction_failures / len(results_df))),
        }

        # Label distribution
        if not results_df["extracted_response"].isnull().all():
            value_counts = results_df["extracted_response"].value_counts()
            stats["label_distribution"] = value_counts.to_dict()

    # Accuracy if ground truth available
    if (
        "ground_truth" in results_df.columns
        and "extracted_response" in results_df.columns
    ):
        valid_mask = (
            results_df["extracted_response"].notna()
            & results_df["ground_truth"].notna()
        )
        valid_df = results_df[valid_mask]

        if len(valid_df) > 0:
            correct = (valid_df["extracted_response"] == valid_df["ground_truth"]).sum()
            stats["accuracy"] = float(correct / len(valid_df))
            stats["evaluated_samples"] = len(valid_df)

    # Prompt statistics
    if "prompt" in results_df.columns:
        prompt_lengths = results_df["prompt"].str.len()
        stats["prompt_stats"] = {
            "mean_length": float(prompt_lengths.mean()),
            "unique_prompts": int(results_df["prompt"].nunique()),
        }

    return stats


def create_error_analysis(
    results_df: pd.DataFrame, output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Create error analysis report for FLAME results.

    Args:
        results_df: Results DataFrame
        output_path: Optional path to save error analysis

    Returns:
        Error analysis DataFrame
    """
    errors = []

    # Check each row for potential errors
    for idx, row in results_df.iterrows():
        error_info = {
            "index": idx,
            "input": row.get("input", "")[:100],  # First 100 chars
        }

        # Check extraction failure
        if pd.isna(row.get("extracted_response")):
            error_info["error_type"] = "extraction_failure"
            error_info["raw_response"] = row.get("raw_response", "")[:200]
            errors.append(error_info)

        # Check mismatch with ground truth
        elif "ground_truth" in row and not pd.isna(row["ground_truth"]):
            if row.get("extracted_response") != row["ground_truth"]:
                error_info["error_type"] = "prediction_error"
                error_info["predicted"] = row["extracted_response"]
                error_info["expected"] = row["ground_truth"]
                error_info["raw_response"] = row.get("raw_response", "")[:200]
                errors.append(error_info)

    error_df = pd.DataFrame(errors)

    # Add summary statistics
    if len(error_df) > 0:
        error_df.attrs["total_errors"] = len(error_df)
        error_df.attrs["error_rate"] = len(error_df) / len(results_df)

        # Error type distribution
        if "error_type" in error_df.columns:
            error_df.attrs["error_types"] = (
                error_df["error_type"].value_counts().to_dict()
            )

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        error_df.to_csv(output_path, index=False)
        logger.info(f"Saved error analysis to: {output_path}")

    return error_df
