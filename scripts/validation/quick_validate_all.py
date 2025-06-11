#!/usr/bin/env python3
"""Quick validation of all FLaME tasks with Ollama (excluding econlogicqa and mmlu)."""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# All tasks except econlogicqa and mmlu (22 total)
TASKS_TO_VALIDATE = [
    # Classification tasks (fastest)
    ("banking77", 5, 32),  # Intent classification
    ("fpb", 5, 32),  # Sentiment analysis
    ("fomc", 5, 32),  # Hawkish/dovish classification
    ("headlines", 5, 32),  # Sentiment analysis
    ("numclaim", 5, 32),  # Claim classification
    ("causal_classification", 5, 32),  # Causal classification
    ("causal_detection", 5, 32),  # Causal detection
    # NER/Extraction tasks (medium)
    ("finer", 5, 50),  # NER task
    ("finentity", 5, 50),  # Entity extraction
    ("finred", 5, 50),  # Relation extraction
    # QA tasks (slower)
    ("fiqa_task1", 3, 64),  # QA sentiment
    ("fiqa_task2", 3, 64),  # QA classification
    ("finqa", 3, 64),  # Financial QA
    ("tatqa", 3, 64),  # Table QA
    ("convfinqa", 2, 64),  # Conversational QA
    ("subjectiveqa", 3, 64),  # Subjective QA
    # Multiple choice (medium)
    ("bizbench", 3, 50),  # Business benchmark
    ("finbench", 3, 50),  # Financial benchmark
    ("fnxl", 3, 50),  # Financial explanation
    ("refind", 3, 50),  # Relationship extraction
    # Summarization (slowest)
    ("ectsum", 2, 128),  # ECTSum summarization
    ("edtsum", 2, 128),  # EDTSum summarization
]


def run_task(task_name, batch_size, max_tokens):
    """Run inference and evaluation for a task."""
    print(f"\n{'=' * 60}")
    print(f"Validating {task_name.upper()}")
    print(f"{'=' * 60}")

    # Run inference
    inference_cmd = f"uv run python main.py --config configs/development.yaml --mode inference --tasks {task_name} --batch_size {batch_size} --max_tokens {max_tokens}"

    print("\n🚀 Running inference...")
    start_time = time.time()
    result = subprocess.run(inference_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Inference failed: {result.stderr}")
        return False

    inference_time = time.time() - start_time
    print(f"✅ Inference completed in {inference_time:.1f}s")

    # Find the latest results file
    results_dir = Path(f"results/{task_name}/ollama")
    if not results_dir.exists():
        print("❌ No results directory found")
        return False

    csv_files = list(results_dir.glob("*.csv"))
    if not csv_files:
        print("❌ No results files found")
        return False

    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"📄 Results: {latest_file.name}")

    # Run evaluation
    eval_cmd = f'uv run python main.py --config configs/development.yaml --mode evaluate --tasks {task_name} --file_name "{latest_file}"'

    print("\n📊 Running evaluation...")
    eval_start = time.time()
    eval_result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)

    if eval_result.returncode != 0:
        print(f"❌ Evaluation failed: {eval_result.stderr}")
        return False

    eval_time = time.time() - eval_start
    print(f"✅ Evaluation completed in {eval_time:.1f}s")

    # Check for evaluation results
    eval_dir = Path(f"evaluations/{task_name}/ollama")
    if eval_dir.exists():
        eval_files = list(eval_dir.glob("*_metrics.csv"))
        if eval_files:
            latest_eval = max(eval_files, key=lambda p: p.stat().st_mtime)
            print(f"📊 Metrics: {latest_eval.name}")

    return True


def main():
    """Run all validations."""
    print("🔧 FLaME Complete Task Validation with Ollama")
    print("=" * 80)
    print("Model: ollama/qwen2.5:1.5b")
    print("Purpose: Validate all 22 tasks functionality (not performance)")
    print(f"Tasks to validate: {len(TASKS_TO_VALIDATE)}")
    print("=" * 80)

    successful = []
    failed = []

    for task_name, batch_size, max_tokens in TASKS_TO_VALIDATE:
        try:
            if run_task(task_name, batch_size, max_tokens):
                successful.append(task_name)
                print(f"\n✅ {task_name} validation PASSED")
            else:
                failed.append(task_name)
                print(f"\n❌ {task_name} validation FAILED")
        except Exception as e:
            print(f"\n💥 {task_name} crashed: {str(e)}")
            failed.append(task_name)

        # Small delay between tasks
        time.sleep(2)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(successful)}/{len(TASKS_TO_VALIDATE)}")
    if successful:
        for task in successful:
            print(f"   - {task}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for task in failed:
            print(f"   - {task}")

    # Save validation results
    results = []
    for task_name, _, _ in TASKS_TO_VALIDATE:
        status = "✅ Success" if task_name in successful else "❌ Failed"
        results.append({"task": task_name, "status": status})

    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"validation_results_{timestamp}.csv"
    df_results.to_csv(output_file, index=False)

    print(f"\n📄 Results saved to: {output_file}")
    print("\n📝 Next steps:")
    print("1. Update task_validation_tracker.md with results")
    print("2. For production validation, use Together AI model")
    print("3. Check logs for any issues")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
