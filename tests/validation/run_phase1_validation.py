#!/usr/bin/env python3
"""
Phase 1 Validation Runner
========================
Simple script to validate both FOMC implementations work correctly.
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime


def run_test(implementation: str, num_samples: int = 3):
    """Run a test with specified implementation."""

    datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 60}")
    print(f"Testing {implementation.upper()} Implementation")
    print(f"Samples: {num_samples}")
    print(f"{'=' * 60}\n")

    # Prepare command based on implementation
    if implementation == "native":
        cmd = [
            "uv",
            "run",
            "python",
            "main.py",
            "--mode",
            "inference",
            "--task",
            "fomc",  # Changed from --dataset to --task
            "--model",
            "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "--max_tokens",
            "20",
            "--temperature",
            "0.0",
            "--num_samples",
            str(num_samples),
        ]
    else:  # benchforge
        cmd = [
            "uv",
            "run",
            "python",
            "src/flame/main_benchforge.py",  # Correct path
            "--mode",
            "inference",
            "--task",
            "fomc",
            "--model",
            "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "--max_tokens",
            "20",
            "--temperature",
            "0.0",
            "--num_samples",
            str(num_samples),
        ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )
        elapsed = time.time() - start_time

        print(f"\nCompleted in {elapsed:.2f} seconds")

        # Check for errors
        if result.returncode != 0:
            print(f"❌ Error running {implementation}:")
            print(result.stderr[-500:] if result.stderr else "No error output")
            return None

        # Parse output for results path
        output_path = None
        for line in result.stdout.split("\n"):
            if "Results saved to" in line or "results saved to" in line:
                # Extract path from line
                parts = line.split("to")[-1].strip()
                parts = parts.replace(":", "").strip()
                if parts:
                    output_path = parts
                    break

        if output_path:
            print(f"✅ Results saved to: {output_path}")
        else:
            print("⚠️ Could not find results path in output")
            print("Output snippet:", result.stdout[-500:])

        return {
            "implementation": implementation,
            "success": True,
            "output_path": output_path,
            "elapsed_time": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        print(f"❌ Timeout running {implementation}")
        return None
    except Exception as e:
        print(f"❌ Exception running {implementation}: {e}")
        return None


def main():
    """Main validation runner."""

    print("\n" + "🚀 " * 20)
    print("PHASE 1 VALIDATION - FOMC IMPLEMENTATION COMPARISON")
    print("🚀 " * 20)

    num_samples = 3  # Small number for quick validation

    # Test both implementations
    results = {}

    # Test native
    print("\n1️⃣ Testing Native FLAME Implementation")
    native_result = run_test("native", num_samples)
    results["native"] = native_result

    # Test BenchForge
    print("\n2️⃣ Testing BenchForge Implementation")
    benchforge_result = run_test("benchforge", num_samples)
    results["benchforge"] = benchforge_result

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    both_work = (
        results["native"] is not None
        and results["native"]["success"]
        and results["benchforge"] is not None
        and results["benchforge"]["success"]
    )

    if both_work:
        print("\n✅ PHASE 1 VALIDATION PASSED!")
        print("Both implementations are working correctly.")

        if results["native"]["output_path"] and results["benchforge"]["output_path"]:
            print("\nOutput files:")
            print(f"  Native: {results['native']['output_path']}")
            print(f"  BenchForge: {results['benchforge']['output_path']}")
            print(
                "\nYou can manually compare these files to verify output consistency."
            )

        print("\n✅ Ready to proceed to Phase 2 migration!")

    else:
        print("\n❌ VALIDATION FAILED")
        if not results["native"] or not results["native"]["success"]:
            print("  - Native implementation failed")
        if not results["benchforge"] or not results["benchforge"]["success"]:
            print("  - BenchForge implementation failed")
        print("\nPlease review the errors above and fix before proceeding.")

    # Save results
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"phase1_validation_{timestamp}.json"

    # Prepare report
    report = {
        "timestamp": timestamp,
        "num_samples": num_samples,
        "both_implementations_work": both_work,
        "results": {
            "native": {
                "success": results["native"]["success"] if results["native"] else False,
                "output_path": results["native"]["output_path"]
                if results["native"]
                else None,
                "elapsed_time": results["native"]["elapsed_time"]
                if results["native"]
                else None,
            }
            if results["native"]
            else {"success": False},
            "benchforge": {
                "success": results["benchforge"]["success"]
                if results["benchforge"]
                else False,
                "output_path": results["benchforge"]["output_path"]
                if results["benchforge"]
                else None,
                "elapsed_time": results["benchforge"]["elapsed_time"]
                if results["benchforge"]
                else None,
            }
            if results["benchforge"]
            else {"success": False},
        },
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📊 Report saved to: {report_path}")

    return 0 if both_work else 1


if __name__ == "__main__":
    sys.exit(main())
