#!/usr/bin/env python3
"""
A/B Testing for FOMC Implementation Migration.
Compares native FLAME and BenchForge implementations.
"""

import subprocess
import pandas as pd
from pathlib import Path
import json
import sys
import time
import argparse
from typing import Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_inference(
    implementation: str,
    task: str = "fomc",
    num_samples: int = 10,
    model: Optional[str] = None,
) -> Path:
    """Run inference with specified implementation.

    Args:
        implementation: 'native' or 'benchforge'
        task: Task name (default: 'fomc')
        num_samples: Number of samples to process
        model: Model to use (optional)

    Returns:
        Path to results file
    """
    logger.info(f"Running {implementation} implementation for task '{task}'...")

    # Build command
    cmd = ["uv", "run", "python", "main.py", "--mode", "inference", "--tasks", task]

    # Add implementation flag
    if implementation == "native":
        cmd.append("--use-native")
    elif implementation == "benchforge":
        cmd.append("--use-benchforge")
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    # Add model if specified
    if model:
        cmd.extend(["--model", model])
    else:
        # Use default model
        cmd.extend(["--model", "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"])

    # Add other parameters for consistency
    cmd.extend(["--max_tokens", "20", "--temperature", "0.0", "--batch_size", "10"])

    # TODO: Add num_samples support when available
    # For now, we'll process all samples and truncate later

    logger.debug(f"Command: {' '.join(cmd)}")

    # Run command
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,  # Project root
    )
    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Failed to run {implementation} inference")

    # Parse output to get results file path
    results_path = None
    for line in result.stdout.split("\n"):
        if "Results saved to" in line or "results saved to" in line:
            # Extract path from line
            parts = line.split("to")[-1].strip()
            parts = parts.replace(":", "").strip()
            if parts and Path(parts).exists():
                results_path = Path(parts)
                break

    if not results_path:
        # Try to find the most recent results file
        results_dir = Path("results") / task
        if results_dir.exists():
            result_files = list(results_dir.glob("*.csv"))
            if result_files:
                results_path = max(result_files, key=lambda p: p.stat().st_mtime)
                logger.warning(
                    f"Could not parse output path, using most recent: {results_path}"
                )

    if not results_path or not results_path.exists():
        logger.error("Could not find results file")
        logger.debug(f"STDOUT: {result.stdout[-1000:]}")  # Last 1000 chars
        raise RuntimeError(f"Could not find results file for {implementation}")

    logger.info(f"  Completed in {elapsed:.2f}s")
    logger.info(f"  Results: {results_path}")

    return results_path


def compare_results(
    native_path: Path, benchforge_path: Path, num_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Compare results from both implementations.

    Args:
        native_path: Path to native results
        benchforge_path: Path to BenchForge results
        num_samples: Limit comparison to this many samples

    Returns:
        Dictionary with comparison metrics
    """
    logger.info("Comparing results...")

    # Load results
    native_df = pd.read_csv(native_path)
    benchforge_df = pd.read_csv(benchforge_path)

    # Normalize column names using migration utils
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from flame.utils.migration_utils import (
        normalize_results,
        compare_results as compare_dfs,
    )

    native_df = normalize_results(native_df, "native")
    benchforge_df = normalize_results(benchforge_df, "benchforge")

    # Limit to num_samples if specified
    if num_samples:
        native_df = native_df.head(num_samples)
        benchforge_df = benchforge_df.head(num_samples)

    # Use utility function for comparison
    comparison = compare_dfs(native_df, benchforge_df)

    # Add timing information if available
    comparison["native_path"] = str(native_path)
    comparison["benchforge_path"] = str(benchforge_path)

    return comparison


def run_ab_test(
    task: str = "fomc",
    num_samples: int = 20,
    model: Optional[str] = None,
    save_report: bool = True,
) -> Dict[str, Any]:
    """Run complete A/B test.

    Args:
        task: Task to test
        num_samples: Number of samples to test
        model: Model to use
        save_report: Whether to save report to file

    Returns:
        Test results dictionary
    """
    print("\n" + "=" * 60)
    print(f"A/B Test: {task.upper()} Implementation Comparison")
    print("=" * 60)

    results = {
        "task": task,
        "num_samples": num_samples,
        "model": model or "default",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        # Run native implementation
        print("\n1️⃣  Running native implementation...")
        native_start = time.time()
        native_path = run_inference("native", task, num_samples, model)
        native_time = time.time() - native_start
        results["native_time"] = native_time

        # Run BenchForge implementation
        print("\n2️⃣  Running BenchForge implementation...")
        bf_start = time.time()
        benchforge_path = run_inference("benchforge", task, num_samples, model)
        bf_time = time.time() - bf_start
        results["benchforge_time"] = bf_time

        # Compare results
        print("\n3️⃣  Comparing results...")
        comparison = compare_results(native_path, benchforge_path, num_samples)
        results["comparison"] = comparison

        # Print summary
        print("\n" + "=" * 60)
        print("A/B TEST RESULTS")
        print("=" * 60)
        print(f"Task: {task}")
        print(f"Samples tested: {comparison.get('num_samples_1', 'N/A')}")
        print("\nPerformance:")
        print(f"  Native time: {native_time:.2f}s")
        print(f"  BenchForge time: {bf_time:.2f}s")
        print(f"  Speed ratio (BF/Native): {bf_time / native_time:.2f}x")

        if "extraction_match_rate" in comparison:
            print("\nAccuracy:")
            print(f"  Extraction match rate: {comparison['extraction_match_rate']:.2%}")
            print(
                f"  Matches: {comparison.get('matches', 0)}/{comparison.get('num_samples_1', 0)}"
            )

            if comparison["extraction_match_rate"] >= 0.95:
                print("\n✅ Implementations are producing consistent results!")
                results["status"] = "PASSED"
            else:
                print("\n⚠️ Differences detected between implementations")
                results["status"] = "DIFFERENCES"

                if comparison.get("differences"):
                    print("\nFirst 5 differences:")
                    for diff in comparison["differences"][:5]:
                        print(
                            f"  Sample {diff['index']}: "
                            f"native={diff.get('df1_label', 'N/A')}, "
                            f"benchforge={diff.get('df2_label', 'N/A')}"
                        )
        else:
            print("\n⚠️ Could not compare extraction results")
            results["status"] = "INCOMPLETE"

        # Save report if requested
        if save_report:
            report_dir = Path("ab_test_reports")
            report_dir.mkdir(exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"ab_test_{task}_{timestamp}.json"

            with open(report_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"\n📊 Detailed report saved to: {report_path}")

        return results

    except Exception as e:
        logger.error(f"A/B test failed: {e}")
        results["status"] = "FAILED"
        results["error"] = str(e)
        return results


def main():
    """Main entry point for A/B testing."""
    parser = argparse.ArgumentParser(
        description="A/B Testing for FLAME Implementation Migration"
    )

    parser.add_argument(
        "--task", type=str, default="fomc", help="Task to test (default: fomc)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to test (default: 20)",
    )
    parser.add_argument("--model", type=str, help="Model to use for testing")
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save test report"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run A/B test
    results = run_ab_test(
        task=args.task,
        num_samples=args.num_samples,
        model=args.model,
        save_report=not args.no_save,
    )

    # Exit with appropriate code
    if results.get("status") == "PASSED":
        sys.exit(0)
    elif results.get("status") == "FAILED":
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
