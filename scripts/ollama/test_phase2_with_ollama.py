#!/usr/bin/env python3
"""Test Phase 2 validation tasks with Ollama for quick iteration."""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd):
    """Run a command and return success status."""
    print(f"\n🚀 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Success!")
        return True
    else:
        print(f"❌ Failed with error:\n{result.stderr}")
        return False


def test_task(task_name, run_inference=True, run_evaluation=True):
    """Test a specific task with Ollama."""
    print(f"\n{'=' * 60}")
    print(f"Testing {task_name.upper()} with Ollama")
    print(f"{'=' * 60}")

    # Run inference
    if run_inference:
        print(f"\n📝 Running inference for {task_name}...")
        inference_cmd = f"uv run python main.py --config configs/development.yaml --mode inference --tasks {task_name} --batch_size 5 --max_tokens 128"

        start_time = time.time()
        success = run_command(inference_cmd)
        elapsed = time.time() - start_time

        if success:
            print(f"⏱️  Inference completed in {elapsed:.1f} seconds")

            # Find the latest results file
            results_dir = Path(f"results/{task_name}/ollama")
            if results_dir.exists():
                latest_file = max(
                    results_dir.glob("*.csv"),
                    key=lambda p: p.stat().st_mtime,
                    default=None,
                )
                if latest_file:
                    print(f"📄 Results saved to: {latest_file}")

                    # Run evaluation if requested
                    if run_evaluation:
                        print(f"\n📊 Running evaluation for {task_name}...")
                        eval_cmd = f'uv run python main.py --config configs/development.yaml --mode evaluate --tasks {task_name} --file_name "{latest_file}"'

                        eval_start = time.time()
                        eval_success = run_command(eval_cmd)
                        eval_elapsed = time.time() - eval_start

                        if eval_success:
                            print(
                                f"⏱️  Evaluation completed in {eval_elapsed:.1f} seconds"
                            )

                            # Find evaluation results
                            eval_dir = Path(f"evaluations/{task_name}/ollama")
                            if eval_dir.exists():
                                latest_eval = max(
                                    eval_dir.glob("*.csv"),
                                    key=lambda p: p.stat().st_mtime,
                                    default=None,
                                )
                                if latest_eval:
                                    print(f"📊 Evaluation saved to: {latest_eval}")
                        else:
                            print(f"⚠️  Evaluation failed for {task_name}")
                else:
                    print("⚠️  No results file found")
        else:
            print(f"⚠️  Inference failed for {task_name}")

    return success


def main():
    """Main test runner for Phase 2 tasks."""
    print("🔧 FLaME Phase 2 Validation with Ollama")
    print("=" * 80)
    print("Using local Ollama model for quick testing")
    print("Model: ollama/qwen2.5:1.5b")
    print("=" * 80)

    # Phase 2 priority tasks
    phase2_tasks = [
        ("causal_detection", True, True),  # Fixed TQDM logging issues
        ("convfinqa", True, True),  # Optimized batch processing
        ("tatqa", True, True),  # Validated batch processing
        ("finqa", True, True),  # Registered and validated
        ("fnxl", False, True),  # JSON parsing fixed, test eval only
    ]

    # Quick win tasks would be:
    # ("ectsum", True, True),  # BERTScore evaluation
    # ("edtsum", True, True),  # BERTScore evaluation

    print("\n📋 Testing Phase 2 Priority Tasks:")
    for task, inference, evaluation in phase2_tasks[:2]:  # Start with first 2
        print(f"  - {task}")

    # Ask user which tasks to run
    print("\nWhich tasks would you like to test?")
    print("1. causal_detection (Phase 2 fix)")
    print("2. convfinqa (Phase 2 optimization)")
    print("3. Both")
    print("4. Quick wins (ectsum, edtsum)")
    print("5. Exit")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        test_task("causal_detection", True, True)
    elif choice == "2":
        test_task("convfinqa", True, True)
    elif choice == "3":
        test_task("causal_detection", True, True)
        test_task("convfinqa", True, True)
    elif choice == "4":
        test_task("ectsum", True, True)
        test_task("edtsum", True, True)
    elif choice == "5":
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice")
        sys.exit(1)

    print("\n✅ Phase 2 validation testing complete!")
    print("\nNote: These are test runs with Ollama. For production validation:")
    print("  - Use Together AI model with full config")
    print("  - Run with larger batch sizes")
    print("  - Verify all metrics match expected ranges")


if __name__ == "__main__":
    main()
