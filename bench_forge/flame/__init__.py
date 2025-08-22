"""FLAME integration module for BenchForge.

This module makes FLAME a first-class citizen of BenchForge,
providing specialized adapters and utilities for financial LLM evaluation.
"""

from bench_forge.flame.adapter import (
    FLAMEAdapter,
    FLAMETask,
    FLAMEConfig,
    flame_task,
)

from bench_forge.flame.utils import (
    args_to_config,
    load_flame_dataset,
    process_flame_results,
)

__all__ = [
    "FLAMEAdapter",
    "FLAMETask",
    "FLAMEConfig",
    "flame_task",
    "args_to_config",
    "load_flame_dataset",
    "process_flame_results",
]
