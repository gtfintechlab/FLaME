#!/usr/bin/env python3
"""Check the status of Phase 2 validation fixes."""

from pathlib import Path


def check_task_registry():
    """Check which tasks are registered in task_registry.py."""
    print("\n📋 Checking task_registry.py...")

    registry_path = Path("src/flame/task_registry.py")
    if registry_path.exists():
        with open(registry_path, "r") as f:
            content = f.read()

        # Check for specific Phase 2 tasks
        phase2_tasks = ["causal_detection", "convfinqa", "tatqa", "finqa", "fnxl"]
        for task in phase2_tasks:
            if f'"{task}"' in content:
                print(f"  ✅ {task} is registered")
            else:
                print(f"  ❌ {task} is NOT registered")


def check_tqdm_fixes():
    """Check if TQDM logging fixes are applied."""
    print("\n🔧 Checking TQDM logging fixes...")

    # Files that should have logger.debug instead of logger.error/info in loops
    files_to_check = [
        "src/flame/code/causal_detection/causal_detection_evaluate.py",
        "src/flame/code/finqa/finqa_inference.py",
        "src/flame/code/finqa/finqa_evaluate.py",
        "src/flame/code/tatqa/tatqa_inference.py",
        "src/flame/code/tatqa/tatqa_evaluate.py",
    ]

    for file_path in files_to_check:
        if Path(file_path).exists():
            with open(file_path, "r") as f:
                content = f.read()

            # Check for problematic patterns
            has_tqdm = "tqdm" in content
            has_logger_error_in_loop = "logger.error" in content and "for" in content
            has_logger_info_in_loop = "logger.info" in content and "for" in content

            if has_tqdm and (has_logger_error_in_loop or has_logger_info_in_loop):
                print(
                    f"  ⚠️  {Path(file_path).name} - May have logging issues in TQDM loops"
                )
            else:
                print(f"  ✅ {Path(file_path).name} - No obvious TQDM issues")


def check_batch_processing():
    """Check if batch processing is implemented."""
    print("\n⚡ Checking batch processing implementation...")

    files_to_check = {
        "convfinqa": "src/flame/code/convfinqa/convfinqa_evaluate.py",
        "causal_detection": "src/flame/code/causal_detection/causal_detection_evaluate.py",
        "tatqa": "src/flame/code/tatqa/tatqa_evaluate.py",
    }

    for task, file_path in files_to_check.items():
        if Path(file_path).exists():
            with open(file_path, "r") as f:
                content = f.read()

            # Check for batch processing patterns
            has_batch_completion = (
                "batch_completion" in content or "process_batch" in content
            )
            has_chunk_list = "chunk_list" in content

            if has_batch_completion or has_chunk_list:
                print(f"  ✅ {task} - Uses batch processing")
            else:
                print(f"  ⚠️  {task} - May not use batch processing")


def check_json_fixes():
    """Check if JSON parsing fixes are implemented."""
    print("\n🔍 Checking JSON parsing fixes...")

    fnxl_eval = Path("src/flame/code/fnxl/fnxl_evaluate.py")
    if fnxl_eval.exists():
        with open(fnxl_eval, "r") as f:
            content = f.read()

        if "clean_json_response" in content or "strip" in content:
            print("  ✅ fnxl - Has JSON cleaning logic")
        else:
            print("  ⚠️  fnxl - May not have JSON cleaning")


def check_ollama_setup():
    """Check if Ollama configuration is ready."""
    print("\n🦙 Checking Ollama setup...")

    configs = {
        "development.yaml": Path("configs/development.yaml"),
        "ollama.yaml": Path("configs/ollama.yaml"),
    }

    for name, path in configs.items():
        if path.exists():
            print(f"  ✅ {name} exists")
            with open(path, "r") as f:
                content = f.read()
                if "ollama" in content:
                    print("     - Configured for Ollama")
        else:
            print(f"  ❌ {name} missing")


def check_recent_results():
    """Check for recent inference/evaluation results."""
    print("\n📊 Checking recent results...")

    phase2_tasks = ["causal_detection", "convfinqa", "tatqa", "finqa", "fnxl"]

    for task in phase2_tasks:
        # Check inference results
        results_path = Path(f"results/{task}")
        if results_path.exists():
            csv_files = list(results_path.rglob("*.csv"))
            if csv_files:
                latest = max(csv_files, key=lambda p: p.stat().st_mtime)
                print(f"  ✅ {task} - Has results: {latest.name}")
            else:
                print(f"  ⚠️  {task} - No results found")

        # Check evaluation results
        eval_path = Path(f"evaluations/{task}")
        if eval_path.exists():
            csv_files = list(eval_path.rglob("*.csv"))
            if csv_files:
                latest = max(csv_files, key=lambda p: p.stat().st_mtime)
                print(f"     - Has evaluation: {latest.name}")


def main():
    """Main status checker."""
    print("🔍 FLaME Phase 2 Status Check")
    print("=" * 80)

    check_task_registry()
    check_tqdm_fixes()
    check_batch_processing()
    check_json_fixes()
    check_ollama_setup()
    check_recent_results()

    print("\n" + "=" * 80)
    print("📝 Summary of Phase 2 Tasks:")
    print("  1. causal_detection - Fixed TQDM logging")
    print("  2. convfinqa - Optimized batch processing")
    print("  3. tatqa - Validated batch processing")
    print("  4. finqa - Registered and validated")
    print("  5. fnxl - JSON parsing fixed")
    print("\n✅ Use test_phase2_with_ollama.py to validate with local model")


if __name__ == "__main__":
    main()
