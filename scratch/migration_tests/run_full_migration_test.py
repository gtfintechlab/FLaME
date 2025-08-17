#!/usr/bin/env python3
"""
Full End-to-End Migration Test for FOMC
Tests both native FLAME and BenchForge with ALL datapoints using live API
"""

import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def log_message(msg, level="INFO"):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def run_native_inference(max_samples=None):
    """Run native FLAME inference."""
    log_message("Starting NATIVE FLAME inference...")

    cmd = [
        "uv",
        "run",
        "python",
        "main.py",
        "--config",
        "configs/default.yaml",
        "--mode",
        "inference",
        "--tasks",
        "fomc",
        "--model",
        "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "--use-native",  # Force native implementation
    ]

    if max_samples:
        cmd.extend(["--max_tokens", str(max_samples)])

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time

    if result.returncode != 0:
        log_message(f"Native inference failed: {result.stderr}", "ERROR")
        return None, duration

    # Parse output to find results file
    results_path = None
    for line in result.stdout.split("\n"):
        if "Results saved to" in line or "saved to" in line.lower():
            # Extract path from the line
            parts = line.split("/")
            if "results" in parts:
                idx = parts.index("results")
                results_path = "/".join(parts[idx:]).strip()
                if not results_path.startswith("/"):
                    results_path = Path.cwd() / results_path
                break

    if not results_path:
        # Look for the most recent CSV in results/fomc/
        results_dir = Path("results/fomc")
        if results_dir.exists():
            csv_files = list(results_dir.glob("*.csv"))
            if csv_files:
                results_path = max(csv_files, key=lambda p: p.stat().st_mtime)
                log_message(f"Found results at: {results_path}")

    log_message(f"Native inference completed in {duration:.2f} seconds")
    return results_path, duration


def run_benchforge_inference(max_samples=None):
    """Run BenchForge inference."""
    log_message("Starting BENCHFORGE inference...")

    cmd = [
        "uv",
        "run",
        "python",
        "main.py",
        "--config",
        "configs/default.yaml",
        "--mode",
        "inference",
        "--tasks",
        "fomc",
        "--model",
        "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "--use-benchforge",  # Force BenchForge implementation
    ]

    if max_samples:
        cmd.extend(["--max_tokens", str(max_samples)])

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time

    if result.returncode != 0:
        log_message(f"BenchForge inference failed: {result.stderr}", "ERROR")
        return None, duration

    # Parse output to find results file
    results_path = None
    for line in result.stdout.split("\n"):
        if "Results saved to" in line or "saved to" in line.lower():
            # Extract path from the line
            parts = line.split("/")
            if "results" in parts:
                idx = parts.index("results")
                results_path = "/".join(parts[idx:]).strip()
                if not results_path.startswith("/"):
                    results_path = Path.cwd() / results_path
                break

    if not results_path:
        # Look for the most recent CSV in results/fomc/
        results_dir = Path("results/fomc")
        if results_dir.exists():
            csv_files = list(results_dir.glob("*.csv"))
            if csv_files:
                results_path = max(csv_files, key=lambda p: p.stat().st_mtime)
                log_message(f"Found results at: {results_path}")

    log_message(f"BenchForge inference completed in {duration:.2f} seconds")
    return results_path, duration


def compare_results(native_path, benchforge_path):
    """Compare results from both implementations."""
    log_message("Comparing results...")

    # Load DataFrames
    native_df = (
        pd.read_csv(native_path)
        if isinstance(native_path, (str, Path))
        else native_path
    )
    benchforge_df = (
        pd.read_csv(benchforge_path)
        if isinstance(benchforge_path, (str, Path))
        else benchforge_path
    )

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "native_samples": len(native_df),
        "benchforge_samples": len(benchforge_df),
        "metrics": {},
        "differences": [],
        "column_mapping": {},
    }

    # Map columns between implementations
    column_mappings = [
        ("llm_responses", "raw_response"),
        ("extracted_labels", "extracted_response"),
        ("actual_labels", "ground_truth"),
        ("prompts", "prompt"),
        ("inputs", "input"),
    ]

    # Find which columns exist
    for native_col, bf_col in column_mappings:
        if native_col in native_df.columns:
            comparison["column_mapping"][native_col] = "found"
        if bf_col in benchforge_df.columns:
            comparison["column_mapping"][bf_col] = "found"

    # Normalize column names for comparison
    native_normalized = native_df.copy()
    benchforge_normalized = benchforge_df.copy()

    # Map BenchForge columns to native names
    for native_col, bf_col in column_mappings:
        if bf_col in benchforge_normalized.columns:
            benchforge_normalized[native_col] = benchforge_normalized[bf_col]
        if (
            native_col not in native_normalized.columns
            and bf_col in native_normalized.columns
        ):
            native_normalized[native_col] = native_normalized[bf_col]

    # Compare extracted labels
    extract_col = (
        "extracted_labels"
        if "extracted_labels" in native_normalized
        else "extracted_response"
    )

    if extract_col in native_normalized and extract_col in benchforge_normalized:
        # Ensure same length for comparison
        min_len = min(len(native_normalized), len(benchforge_normalized))

        native_labels = native_normalized[extract_col].iloc[:min_len]
        benchforge_labels = benchforge_normalized[extract_col].iloc[:min_len]

        # Calculate match rate
        matches = (native_labels == benchforge_labels).sum()
        comparison["metrics"]["extraction_match_rate"] = matches / min_len
        comparison["metrics"]["extraction_match_count"] = int(matches)

        # Find differences
        for i in range(min_len):
            if native_labels.iloc[i] != benchforge_labels.iloc[i]:
                comparison["differences"].append(
                    {
                        "index": i,
                        "native": str(native_labels.iloc[i]),
                        "benchforge": str(benchforge_labels.iloc[i]),
                    }
                )

    # Compare accuracy if ground truth available
    truth_col = (
        "actual_labels" if "actual_labels" in native_normalized else "ground_truth"
    )

    if truth_col in native_normalized and extract_col in native_normalized:
        native_correct = (
            native_normalized[extract_col] == native_normalized[truth_col]
        ).sum()
        native_accuracy = native_correct / len(native_normalized)
        comparison["metrics"]["native_accuracy"] = native_accuracy
        comparison["metrics"]["native_correct"] = int(native_correct)

    if truth_col in benchforge_normalized and extract_col in benchforge_normalized:
        benchforge_correct = (
            benchforge_normalized[extract_col] == benchforge_normalized[truth_col]
        ).sum()
        benchforge_accuracy = benchforge_correct / len(benchforge_normalized)
        comparison["metrics"]["benchforge_accuracy"] = benchforge_accuracy
        comparison["metrics"]["benchforge_correct"] = int(benchforge_correct)

    # Calculate extraction success rates
    if extract_col in native_normalized:
        native_extracted = native_normalized[extract_col].notna().sum()
        comparison["metrics"]["native_extraction_rate"] = native_extracted / len(
            native_normalized
        )

    if extract_col in benchforge_normalized:
        benchforge_extracted = benchforge_normalized[extract_col].notna().sum()
        comparison["metrics"]["benchforge_extraction_rate"] = (
            benchforge_extracted / len(benchforge_normalized)
        )

    return comparison


def generate_report(comparison, native_duration, benchforge_duration):
    """Generate comprehensive migration report."""

    report = []
    report.append("=" * 80)
    report.append("FULL MIGRATION TEST REPORT - FOMC")
    report.append("=" * 80)
    report.append(f"Timestamp: {comparison['timestamp']}")
    report.append("")

    # Sample counts
    report.append("SAMPLE COUNTS:")
    report.append(f"  Native samples: {comparison['native_samples']}")
    report.append(f"  BenchForge samples: {comparison['benchforge_samples']}")
    report.append("")

    # Performance
    report.append("PERFORMANCE METRICS:")
    report.append(f"  Native duration: {native_duration:.2f} seconds")
    report.append(f"  BenchForge duration: {benchforge_duration:.2f} seconds")
    report.append(f"  Performance ratio: {benchforge_duration / native_duration:.2f}x")

    if benchforge_duration <= 2 * native_duration:
        report.append("  ✅ Performance within 2x target")
    else:
        report.append("  ⚠️ Performance exceeds 2x target")
    report.append("")

    # Accuracy metrics
    report.append("ACCURACY METRICS:")
    metrics = comparison["metrics"]

    if "native_accuracy" in metrics:
        report.append(
            f"  Native accuracy: {metrics['native_accuracy']:.2%} ({metrics['native_correct']}/{comparison['native_samples']})"
        )

    if "benchforge_accuracy" in metrics:
        report.append(
            f"  BenchForge accuracy: {metrics['benchforge_accuracy']:.2%} ({metrics['benchforge_correct']}/{comparison['benchforge_samples']})"
        )

    if "extraction_match_rate" in metrics:
        report.append(
            f"  Output consistency: {metrics['extraction_match_rate']:.2%} ({metrics['extraction_match_count']} matches)"
        )

        if metrics["extraction_match_rate"] >= 0.95:
            report.append(
                "  ✅ Implementations producing consistent results (>95% match)"
            )
        else:
            report.append(
                f"  ⚠️ Inconsistencies detected ({len(comparison['differences'])} differences)"
            )

    if "native_extraction_rate" in metrics:
        report.append(
            f"  Native extraction success: {metrics['native_extraction_rate']:.2%}"
        )

    if "benchforge_extraction_rate" in metrics:
        report.append(
            f"  BenchForge extraction success: {metrics['benchforge_extraction_rate']:.2%}"
        )
    report.append("")

    # Differences
    if comparison["differences"]:
        report.append("SAMPLE DIFFERENCES (first 10):")
        for diff in comparison["differences"][:10]:
            report.append(
                f"  Sample {diff['index']}: native={diff['native']}, benchforge={diff['benchforge']}"
            )
        if len(comparison["differences"]) > 10:
            report.append(
                f"  ... and {len(comparison['differences']) - 10} more differences"
            )
        report.append("")

    # Migration readiness
    report.append("MIGRATION READINESS ASSESSMENT:")

    ready_items = []
    not_ready_items = []

    # Check criteria
    if comparison["benchforge_samples"] > 0:
        ready_items.append("BenchForge successfully processed samples")
    else:
        not_ready_items.append("BenchForge failed to process samples")

    if benchforge_duration <= 2 * native_duration:
        ready_items.append("Performance within acceptable range")
    else:
        not_ready_items.append("Performance outside acceptable range")

    if metrics.get("extraction_match_rate", 0) >= 0.95:
        ready_items.append("Output consistency >95%")
    else:
        not_ready_items.append("Output consistency <95%")

    if metrics.get("benchforge_extraction_rate", 0) >= 0.95:
        ready_items.append("Extraction success rate >95%")
    else:
        not_ready_items.append("Extraction success rate <95%")

    report.append("✅ Ready:")
    for item in ready_items:
        report.append(f"  - {item}")

    if not_ready_items:
        report.append("❌ Not Ready:")
        for item in not_ready_items:
            report.append(f"  - {item}")
    report.append("")

    # Final recommendation
    report.append("=" * 80)
    report.append("FINAL RECOMMENDATION:")

    if len(ready_items) >= 3 and len(not_ready_items) <= 1:
        report.append("🚀 BENCHFORGE IS READY FOR PRODUCTION MIGRATION")
        report.append("   All critical criteria met. Safe to proceed with migration.")
    else:
        report.append("⚠️ ADDITIONAL WORK NEEDED BEFORE MIGRATION")
        report.append("   Address the issues listed above before proceeding.")

    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Run full migration test."""
    log_message("Starting FULL FOMC Migration Test with ALL datapoints", "INFO")
    log_message("This will run both implementations with live API calls", "WARNING")

    # Check API key
    if not os.getenv("TOGETHER_API_KEY"):
        log_message(
            "TOGETHER_API_KEY not set. Please set it to run live tests.", "ERROR"
        )
        sys.exit(1)

    # Create reports directory
    reports_dir = Path("migration_reports")
    reports_dir.mkdir(exist_ok=True)

    try:
        # Run native inference
        log_message("=" * 60)
        log_message("PHASE 1: Running NATIVE FLAME implementation")
        log_message("=" * 60)
        native_path, native_duration = run_native_inference()

        if not native_path:
            log_message("Native inference failed. Cannot proceed.", "ERROR")
            sys.exit(1)

        log_message(f"Native results saved to: {native_path}")

        # Run BenchForge inference
        log_message("=" * 60)
        log_message("PHASE 2: Running BENCHFORGE implementation")
        log_message("=" * 60)
        benchforge_path, benchforge_duration = run_benchforge_inference()

        if not benchforge_path:
            log_message("BenchForge inference failed. Cannot proceed.", "ERROR")
            sys.exit(1)

        log_message(f"BenchForge results saved to: {benchforge_path}")

        # Compare results
        log_message("=" * 60)
        log_message("PHASE 3: Comparing results")
        log_message("=" * 60)
        comparison = compare_results(native_path, benchforge_path)

        # Generate report
        report = generate_report(comparison, native_duration, benchforge_duration)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"full_migration_test_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        # Save comparison JSON
        json_path = reports_dir / f"comparison_data_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        # Print report
        print("\n" + report)

        log_message(f"Report saved to: {report_path}")
        log_message(f"Comparison data saved to: {json_path}")

        # Update migration monitor
        from flame.utils.migration_monitor import get_migration_monitor

        monitor = get_migration_monitor()

        # Log metrics to monitor
        monitor.log_call("native", "fomc", True, native_duration)
        monitor.log_call("benchforge", "fomc", True, benchforge_duration)

        # Check if we should proceed
        if comparison["metrics"].get("extraction_match_rate", 0) >= 0.95:
            log_message(
                "✅ Migration test PASSED - Implementations are consistent!", "SUCCESS"
            )
            return 0
        else:
            log_message(
                "⚠️ Migration test needs review - Inconsistencies detected", "WARNING"
            )
            return 1

    except Exception as e:
        log_message(f"Test failed: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
