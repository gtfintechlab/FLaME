"""Utilities for working with tests in FLaME.

This module provides helpers for test artifact management, patching, and other
test-specific functionality.
"""

from pathlib import Path
from datetime import date

from flame.config import TEST_OUTPUT_DIR


def get_test_output_path(task: str, filename: str) -> Path:
    """Get a path for test output artifacts that won't be committed.
    
    Args:
        task: Task name (e.g., 'numclaim', 'fpb')
        filename: Base filename for the output
        
    Returns:
        Path object for test output
    """
    # Create task-specific directory
    task_dir = TEST_OUTPUT_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Add date to avoid name collisions
    today = date.today().strftime('%Y%m%d')
    return task_dir / f"{today}_{filename}"
