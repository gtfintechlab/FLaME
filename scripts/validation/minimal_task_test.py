#!/usr/bin/env python3
"""Minimal task validation - Test each task works without running full inference."""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# All local imports after path setup
from src.flame.code.prompts import PromptFormat, get_prompt  # noqa: E402
from src.flame.task_registry import EVALUATE_MAP, INFERENCE_MAP  # noqa: E402
from src.flame.utils.dataset_utils import safe_load_dataset  # noqa: E402

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

# Dataset configurations - skip field check for now, just test loading
DATASET_CONFIGS = {
    "banking77": ("gtfintechlab/banking77", "test"),
    "bizbench": ("gtfintechlab/bizbench", "test"),
    "causal_classification": ("gtfintechlab/causal_classification", "test"),
    "causal_detection": ("gtfintechlab/causal_detection", "test"),
    "convfinqa": ("gtfintechlab/convfinqa", "dev"),  # Uses dev split
    "ectsum": ("gtfintechlab/ECTSum", "test"),
    "edtsum": ("gtfintechlab/EDTSum", "test"),
    "finbench": ("gtfintechlab/finbench", "test"),
    "finentity": (None, None),  # Needs config parameter
    "finer": ("gtfintechlab/finer-ord", "test"),
    "finqa": ("gtfintechlab/finqa", "test"),
    "finred": ("gtfintechlab/REFinD", "test"),
    "fiqa_task1": ("gtfintechlab/fiqa_task1", "test"),
    "fiqa_task2": ("gtfintechlab/fiqa_task2", "test"),
    "fnxl": ("gtfintechlab/fnxl", "test"),
    "fomc": ("gtfintechlab/fomc", "test"),
    "fpb": ("gtfintechlab/fpb", "test"),
    "headlines": ("gtfintechlab/headlines", "test"),
    "numclaim": ("gtfintechlab/numclaim", "test"),
    "refind": ("gtfintechlab/refind", "test"),
    "subjectiveqa": ("gtfintechlab/subjectiveqa", "test"),
    "tatqa": ("gtfintechlab/tatqa", "test"),
}


def test_task_minimal(task: str):
    """Test task with minimal operations."""
    result = {"task": task, "status": "❌", "message": "Not tested"}

    try:
        # 1. Check registration
        if task not in INFERENCE_MAP:
            result["message"] = "Not in inference registry"
            return result

        if task not in EVALUATE_MAP:
            result["message"] = "Not in evaluation registry"
            return result

        # 2. Test dataset loading
        if task in DATASET_CONFIGS:
            dataset_name, split = DATASET_CONFIGS[task]
            if dataset_name is None:
                result["message"] = "Dataset needs config"
            else:
                try:
                    dataset = safe_load_dataset(dataset_name, trust_remote_code=True)
                    test_data = dataset[split]

                    # Check if dataset has data
                    if len(test_data) == 0:
                        result["message"] = "Empty dataset"
                        return result

                    # Just check we can access first item
                    _ = test_data[0]  # Verify we can access data
                    result["message"] = f"Dataset OK ({len(test_data)} examples)"

                except Exception as e:
                    result["message"] = f"Dataset error: {str(e)[:30]}"
                    return result

        # 3. Test prompt generation
        try:
            prompt_func = get_prompt(task, PromptFormat.ZERO_SHOT)
            if prompt_func is None:
                result["message"] = "No prompt function"
                return result

            # Test with dummy text
            test_prompt = prompt_func("test input")
            if not test_prompt:
                result["message"] = "Empty prompt generated"
                return result

        except Exception as e:
            result["message"] = f"Prompt error: {str(e)[:30]}"
            return result

        # If we got here, basic functionality works
        result["status"] = "✅"
        result["message"] = "Ready"

    except Exception as e:
        result["message"] = f"Error: {str(e)[:30]}"

    return result


def main():
    """Run minimal validation."""
    print("\n🔍 FLaME Minimal Task Validation")
    print("=" * 60)
    print("Testing: Registration, Dataset Loading, Prompt Generation")
    print(f"Tasks: {len(TASKS)}")
    print("=" * 60)

    results = []

    for i, task in enumerate(TASKS, 1):
        print(f"\n[{i}/{len(TASKS)}] Testing {task}...", end=" ", flush=True)

        result = test_task_minimal(task)
        results.append(result)

        print(f"{result['status']} {result['message']}")

    # Summary
    success_count = sum(1 for r in results if r["status"] == "✅")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(
        f"Success: {success_count}/{len(TASKS)} ({success_count / len(TASKS) * 100:.0f}%)"
    )

    # Show details
    print("\nTask Status:")
    for r in results:
        print(f"  {r['status']} {r['task']:20} - {r['message']}")

    # Save results
    import pandas as pd

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"minimal_validation_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
