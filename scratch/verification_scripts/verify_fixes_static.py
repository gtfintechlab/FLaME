#!/usr/bin/env python3
"""Static verification that BenchForge fixes are in place."""

import sys
from pathlib import Path


def check_parallel_processing_fix():
    """Check if the parallel processing fix is in place."""
    print("\n1. Checking Parallel Processing Fix...")

    client_file = Path("benchforge/bench_forge/llm/client.py")
    if not client_file.exists():
        print("   ❌ Client file not found")
        return False

    content = client_file.read_text()

    # Check for key indicators of the fix
    checks = [
        ("litellm.batch_completion" in content, "Uses litellm.batch_completion"),
        ("PARALLEL" in content, "Has PARALLEL comment/documentation"),
        ("complete_batch" in content, "Has complete_batch method"),
    ]

    all_good = True
    for check, desc in checks:
        if check:
            print(f"   ✅ {desc}")
        else:
            print(f"   ❌ {desc}")
            all_good = False

    return all_good


def check_extraction_fix():
    """Check if the extraction logic fix is in place."""
    print("\n2. Checking Extraction Logic Fix...")

    fomc_file = Path("benchforge/bench_forge/flame/tasks/fomc.py")
    if not fomc_file.exists():
        print("   ❌ FOMC task file not found")
        return False

    content = fomc_file.read_text()

    # Parse the file to count extraction strategies
    strategies = [
        "startswith",
        "after removing prefix",
        "word boundary",
        "only label present",
        "from line",
        "from pattern",
    ]

    strategy_count = sum(1 for s in strategies if s in content.lower())

    # Check for key features
    checks = [
        (
            strategy_count >= 4,
            f"Has multiple extraction strategies ({strategy_count}/6)",
        ),
        (
            "extract_label_from_response" in content,
            "Has extract_label_from_response method",
        ),
        ("complete_responses" in content, "Stores complete_responses"),
        ("llm_responses" in content, "Stores llm_responses"),
    ]

    all_good = True
    for check, desc in checks:
        if check:
            print(f"   ✅ {desc}")
        else:
            print(f"   ❌ {desc}")
            all_good = False

    return all_good


def check_fallback_extraction():
    """Check if the fallback extraction enhancement exists."""
    print("\n3. Checking Fallback Extraction Enhancement...")

    enhanced_file = Path("benchforge/bench_forge/engine/evaluation_enhanced.py")
    if not enhanced_file.exists():
        print("   ⚠️ Enhanced evaluation not found (optional enhancement)")
        return None  # Not a failure, just not implemented yet

    content = enhanced_file.read_text()

    checks = [
        ("EnhancedEvaluationEngine" in content, "Has EnhancedEvaluationEngine class"),
        (
            "_extract_from_complete_response" in content,
            "Has fallback extraction method",
        ),
        (
            "_prepare_predictions_with_fallback" in content,
            "Has prediction preparation with fallback",
        ),
    ]

    all_good = True
    for check, desc in checks:
        if check:
            print(f"   ✅ {desc}")
        else:
            print(f"   ❌ {desc}")
            all_good = False

    return all_good


def check_backup_files():
    """Check if original files were backed up."""
    print("\n4. Checking Backup Files...")

    backups = [
        ("benchforge/bench_forge/llm/client_original.py", "LLM client backup"),
        ("benchforge/bench_forge/flame/tasks/fomc_original.py", "FOMC task backup"),
    ]

    found_any = False
    for backup_path, desc in backups:
        if Path(backup_path).exists():
            print(f"   ✅ {desc} exists")
            found_any = True
        else:
            print(f"   ⚠️ {desc} not found")

    return found_any


def main():
    """Run all static verification checks."""
    print("=" * 60)
    print("BenchForge Fixes Static Verification")
    print("=" * 60)
    print("\nThis checks if the fixes are in place without running inference.")

    results = {
        "Parallel Processing": check_parallel_processing_fix(),
        "Extraction Logic": check_extraction_fix(),
        "Fallback Extraction": check_fallback_extraction(),
        "Backup Files": check_backup_files(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Count results (None means optional/not critical)
    required_checks = {k: v for k, v in results.items() if v is not None}
    passed = sum(1 for v in required_checks.values() if v)
    total = len(required_checks)

    print(f"\n✅ Passed: {passed}/{total} required checks")

    for check_name, result in results.items():
        if result is None:
            status = "⚠️ Optional"
        elif result:
            status = "✅ Pass"
        else:
            status = "❌ Fail"
        print(f"  {status}: {check_name}")

    success = passed == total

    if success:
        print("\n🎉 All required fixes are in place!")
        print("\nNext steps:")
        print("1. Run a full test when ready (will use API credits)")
        print("2. Monitor extraction rate and performance")
        print("3. Verify results match FLAME output")
    else:
        print("\n⚠️ Some fixes may not be properly applied")
        print("Run: python benchforge_performance_fix.py")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
