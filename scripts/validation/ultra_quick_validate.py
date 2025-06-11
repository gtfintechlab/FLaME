#!/usr/bin/env python3
"""Ultra-quick validation - Test each task with just 1 example to verify functionality."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# All imports after path setup
import litellm  # noqa: E402

from src.flame.task_registry import EVALUATE_MAP, INFERENCE_MAP  # noqa: E402

# Configure litellm
litellm.drop_params = True
litellm.suppress_debug_info = True
os.environ["LITELLM_LOG"] = "ERROR"


# Simple print-based logging for this script
class SimpleLogger:
    def info(self, msg):
        print(msg)

    def error(self, msg):
        print(f"ERROR: {msg}")


logger = SimpleLogger()

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


class MinimalArgs:
    """Minimal args for testing."""

    def __init__(self, task):
        self.dataset = task
        self.model = "ollama/qwen2.5:0.5b"  # Smallest model
        self.max_examples = 1  # Just 1 example!
        self.batch_size = 1
        self.max_tokens = 30  # Very short responses
        self.temperature = 0.1
        self.prompt_format = "zero_shot"
        self.file_name = None
        self.output_dir = str(project_root / "results")
        self.eval_output_dir = str(project_root / "evaluations")
        # Default values for other params
        self.top_p = 1.0
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.stop = None
        self.seed = 42
        self.response_format = None
        self.tools = None
        self.tool_choice = None
        self.parallel_tool_calls = None
        self.drop_params = True
        self.api_version = None
        self.num_retries = 1
        self.retry_delay = 1
        self.use_beam_search = False
        self.best_of = 1
        self.n = 1
        self.stream = False
        self.logprobs = None
        self.echo = False
        self.suffix = None
        self.logit_bias = None
        self.user = None


def test_task(task: str):
    """Test a single task with minimal example."""
    try:
        # Check registration
        if task not in INFERENCE_MAP:
            return "Not registered (inference)"
        if task not in EVALUATE_MAP:
            return "Not registered (evaluation)"

        # Test inference
        args = MinimalArgs(task)
        inference_func = INFERENCE_MAP[task]

        start = time.time()
        df = inference_func(args)

        if df is None or df.empty:
            return "Empty results"

        inference_time = time.time() - start

        # Save minimal results
        timestamp = datetime.now().strftime("%d_%m_%Y_%H%M%S")
        filename = f"{task}_validation_{timestamp}.csv"
        output_path = Path(args.output_dir) / task / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        # Test evaluation
        args.file_name = str(output_path)
        eval_func = EVALUATE_MAP[task]

        eval_start = time.time()
        _ = eval_func(args)  # Run evaluation
        eval_time = time.time() - eval_start

        total_time = inference_time + eval_time
        return f"✅ Success ({total_time:.1f}s)"

    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return "Model not found"
        elif "timeout" in error_msg.lower():
            return "Timeout"
        else:
            return f"Error: {error_msg[:30]}..."


def main():
    """Run ultra-quick validation."""
    print("\n🚀 Ultra-Quick FLaME Validation")
    print("=" * 60)
    print("Model: ollama/qwen2.5:0.5b (smallest)")
    print("Examples per task: 1")
    print("Max tokens: 30")
    print(f"Tasks: {len(TASKS)}")
    print("=" * 60)

    results = []
    start_time = time.time()

    for i, task in enumerate(TASKS, 1):
        print(f"\n[{i}/{len(TASKS)}] {task}... ", end="", flush=True)

        result = test_task(task)
        results.append(
            {
                "task": task,
                "result": result,
                "status": "✅" if result.startswith("✅") else "❌",
            }
        )

        print(result)

    # Summary
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r["status"] == "✅")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Average per task: {total_time / len(TASKS):.1f}s")
    print(
        f"Success rate: {success_count}/{len(TASKS)} ({success_count / len(TASKS) * 100:.0f}%)"
    )

    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ultra_quick_validation_{timestamp}.csv"
    df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")

    # Show failed tasks
    failed = [r for r in results if r["status"] == "❌"]
    if failed:
        print("\nFailed tasks:")
        for r in failed:
            print(f"  - {r['task']}: {r['result']}")


if __name__ == "__main__":
    main()
