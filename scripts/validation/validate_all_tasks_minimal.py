#!/usr/bin/env python3
"""
Validate all FLaME tasks with minimal examples using Ollama.
This script runs both inference and evaluation for each task with only 5 examples.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# All tasks except econlogicqa and mmlu
TASKS = [
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

# Ollama model to use for quick testing
MODEL = "ollama/llama3.2:1b"


def run_task_validation(task: str, max_examples: int = 5):
    """Run inference and evaluation for a single task with limited examples."""
    print(f"\n{'=' * 60}")
    print(f"Validating task: {task}")
    print(f"{'=' * 60}")

    # Create temporary config file with limited examples
    config_content = f"""
model: "{MODEL}"
max_examples: {max_examples}
batch_size: 5
max_tokens: 50
temperature: 0.1
prompt_format: zero_shot
"""

    config_path = project_root / f"configs/temp_{task}.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    try:
        # Run inference
        print(f"\n1. Running inference for {task}...")
        start_time = time.time()

        inference_cmd = [
            "uv",
            "run",
            "python",
            "main.py",
            "--config",
            str(config_path),
            "--mode",
            "inference",
            "--dataset",
            task,
        ]

        result = subprocess.run(
            inference_cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout per task
        )

        if result.returncode != 0:
            print(f"❌ Inference failed for {task}")
            print(f"Error: {result.stderr}")
            return False, "Inference failed"

        inference_time = time.time() - start_time
        print(f"✓ Inference completed in {inference_time:.1f}s")

        # Find the output file
        results_dir = project_root / "results" / task
        if not results_dir.exists():
            print(f"❌ No results directory found for {task}")
            return False, "No results directory"

        # Get the most recent results file
        result_files = list(results_dir.glob(f"{task}_*.csv"))
        if not result_files:
            print(f"❌ No result files found for {task}")
            return False, "No result files"

        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        print(f"Found results file: {latest_file.name}")

        # Check if file has content
        df = pd.read_csv(latest_file)
        print(f"Results shape: {df.shape}")

        # Run evaluation
        print(f"\n2. Running evaluation for {task}...")
        start_time = time.time()

        eval_cmd = [
            "uv",
            "run",
            "python",
            "main.py",
            "--config",
            str(config_path),
            "--mode",
            "evaluate",
            "--dataset",
            task,
            "--file_name",
            str(latest_file),
        ]

        result = subprocess.run(
            eval_cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout per task
        )

        if result.returncode != 0:
            print(f"❌ Evaluation failed for {task}")
            print(f"Error: {result.stderr}")
            return False, "Evaluation failed"

        eval_time = time.time() - start_time
        print(f"✓ Evaluation completed in {eval_time:.1f}s")

        # Check for evaluation results
        eval_dir = project_root / "evaluations" / task
        if eval_dir.exists():
            eval_files = list(eval_dir.glob("*.csv"))
            if eval_files:
                print(f"✓ Evaluation results saved: {len(eval_files)} files")

        return True, "Success"

    except subprocess.TimeoutExpired:
        print(f"⏱️ Task {task} timed out")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Error validating {task}: {str(e)}")
        return False, str(e)
    finally:
        # Clean up temp config
        if config_path.exists():
            config_path.unlink()


def main():
    """Run validation for all tasks."""
    print("FLaME Task Validation with Ollama")
    print(f"Model: {MODEL}")
    print(f"Tasks to validate: {len(TASKS)}")
    print("Examples per task: 5")

    # Check if Ollama is running
    print("\nChecking Ollama connection...")
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
        else:
            print("❌ Ollama is not responding properly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Please ensure Ollama is running: ollama serve")
        return

    # Track results
    results = []
    start_time = time.time()

    # Run validation for each task
    for i, task in enumerate(TASKS, 1):
        print(f"\n[{i}/{len(TASKS)}] Processing {task}")
        success, message = run_task_validation(task)
        results.append({"task": task, "success": success, "message": message})

        # Save intermediate results
        results_df = pd.DataFrame(results)
        results_df.to_csv(project_root / "validation_results.csv", index=False)

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average per task: {total_time / len(TASKS):.1f}s")

    success_count = sum(1 for r in results if r["success"])
    print(
        f"\nSuccess rate: {success_count}/{len(TASKS)} ({success_count / len(TASKS) * 100:.1f}%)"
    )

    # Show failed tasks
    failed = [r for r in results if not r["success"]]
    if failed:
        print("\nFailed tasks:")
        for r in failed:
            print(f"  - {r['task']}: {r['message']}")

    # Save final results
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = project_root / f"validation_results_{timestamp}.csv"
    results_df.to_csv(final_path, index=False)
    print(f"\nResults saved to: {final_path}")


if __name__ == "__main__":
    main()
