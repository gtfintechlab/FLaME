"""Test all tasks work with minimal examples using Ollama."""

import subprocess
import time

import pytest

# All tasks except econlogicqa and mmlu
ALL_TASKS = [
    "banking77",
    "bizbench",
    "causal_classification",
    "causal_detection",
    "convfinqa",
    "ectsum",
    "edtsum",
    "finbench",
    "finentity",
    "finer",
    "finqa",
    "finred",
    "fiqa_task1",
    "fiqa_task2",
    "fnxl",
    "fomc",
    "fpb",
    "headlines",
    "numclaim",
    "refind",
    "subjectiveqa",
    "tatqa",
]


@pytest.mark.parametrize("task", ALL_TASKS)
@pytest.mark.integration
@pytest.mark.requires_ollama
def test_task_inference_minimal(task):
    """Test that each task can start inference successfully."""
    cmd = [
        "python",
        "main.py",
        "--config",
        "configs/development.yaml",
        "--mode",
        "inference",
        "--tasks",
        task,
        "--batch_size",
        "1",
        "--max_tokens",
        "32",
    ]

    # Start process and let it run for 10 seconds
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Give it 10 seconds to start processing
    time.sleep(10)

    # Check if still running (good) or crashed (bad)
    poll = process.poll()

    # Terminate the process
    process.terminate()
    process.wait()

    # If process is None, it was still running (good)
    # If process returned non-zero, it crashed (bad)
    if poll is not None and poll != 0:
        stderr = process.stderr.read()
        pytest.fail(f"Task {task} failed to start: {stderr[:500]}")

    # Task started successfully
    assert True, f"Task {task} started successfully"
