"""FOMC task implementation using BenchForge.

Federal Open Market Committee (FOMC) communication classification task.
"""

from typing import Optional

from flame.benchforge import flame_task

# Import the fixed FOMC implementation from BenchForge
from bench_forge.flame.tasks.fomc import FOMCTask as BenchForgeFOMCTask, FOMCConfig


@flame_task("fomc")
class FOMCTask(BenchForgeFOMCTask):
    """Federal Open Market Committee statement classification task.

    This task classifies FOMC statements as HAWKISH, DOVISH, or NEUTRAL
    based on monetary policy stance.

    This class wraps the fixed BenchForge FOMC implementation with:
    - Proper extraction logic (6 strategies)
    - FLAME-compatible column names
    - Complete response storage for fallback extraction
    """

    def __init__(self, config: Optional[FOMCConfig] = None, **kwargs):
        """Initialize FOMC task with the fixed BenchForge implementation."""
        if config is None:
            config = FOMCConfig(name="fomc")
        elif not isinstance(config, FOMCConfig):
            # Convert generic config to FOMCConfig
            fomc_config = FOMCConfig(
                **config.__dict__ if hasattr(config, "__dict__") else config
            )
            config = fomc_config

        super().__init__(config, **kwargs)

    # All methods inherited from BenchForgeFOMCTask:
    # - create_prompt: Creates zero-shot and few-shot prompts
    # - extract_label_from_response: 6-strategy extraction logic
    # - format_results: Returns FLAME-compatible column names
    # - process_responses: Handles extraction and formatting
