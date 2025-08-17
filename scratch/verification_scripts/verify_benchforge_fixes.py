#!/usr/bin/env python3
"""Quick verification that BenchForge fixes are working."""

import os
import sys
import time
from pathlib import Path

# Load .env file if it exists
if Path(".env").exists():
    from dotenv import load_dotenv

    load_dotenv()

# Set environment variables
os.environ["USE_BENCHFORGE_FOMC"] = "true"

# Try both possible API key names
api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY", "")
if api_key:
    os.environ["TOGETHER_API_KEY"] = api_key
    os.environ["TOGETHERAI_API_KEY"] = api_key


def test_benchforge_with_fixes():
    """Test BenchForge with 5 samples to verify fixes work."""

    print("=" * 60)
    print("Verifying BenchForge Fixes")
    print("=" * 60)

    # Check API key
    if not os.environ.get("TOGETHER_API_KEY"):
        print("ERROR: TOGETHER_API_KEY not set")
        print("Please set TOGETHER_API_KEY or TOGETHERAI_API_KEY environment variable")
        return False

    print("\nRunning BenchForge with full FOMC dataset (496 samples)...")
    print("This will take ~85-100 seconds if fixes are working correctly...")

    # Run BenchForge inference
    # Note: main.py doesn't support --max_samples, will run with full dataset
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
        "--use-benchforge",
    ]

    import subprocess

    start_time = time.time()

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - start_time

    print(f"\nExecution time: {elapsed:.2f} seconds")

    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        print("STDOUT:", result.stdout[-500:] if result.stdout else "None")
        print("STDERR:", result.stderr[-500:] if result.stderr else "None")
        return False

    # Check output for key indicators
    output = result.stdout + result.stderr

    # Look for performance indicators
    if "Processing batch of" in output and "PARALLEL" in output:
        print("✅ Parallel processing detected")
    else:
        print("⚠️ Parallel processing not confirmed")

    # Look for extraction success
    if "Extraction success rate:" in output:
        for line in output.split("\n"):
            if "Extraction success rate:" in line:
                print(f"✅ {line.strip()}")
                break

    # Check if results file was created
    results_dir = Path("results/fomc")
    latest_file = max(results_dir.glob("*.csv"), key=os.path.getctime, default=None)

    if latest_file and (time.time() - os.path.getctime(latest_file)) < 60:
        print(f"✅ Results saved to: {latest_file.name}")

        # Check the file contents
        import pandas as pd

        df = pd.read_csv(latest_file)

        print("\nResults Analysis:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {', '.join(df.columns)}")

        # Check for key columns
        if "complete_responses" in df.columns:
            print("  ✅ Has complete_responses column")
        else:
            print("  ❌ Missing complete_responses column")

        if "extracted_labels" in df.columns:
            extracted_count = df["extracted_labels"].notna().sum()
            success_rate = (extracted_count / len(df)) * 100
            print(
                f"  ✅ Extraction rate: {extracted_count}/{len(df)} ({success_rate:.0f}%)"
            )
        elif "extracted_response" in df.columns:
            extracted_count = df["extracted_response"].notna().sum()
            success_rate = (extracted_count / len(df)) * 100
            print(
                f"  ✅ Extraction rate: {extracted_count}/{len(df)} ({success_rate:.0f}%)"
            )

    return True


if __name__ == "__main__":
    success = test_benchforge_with_fixes()
    if success:
        print("\n✅ Verification complete - fixes appear to be working")
    else:
        print("\n❌ Verification failed - fixes may not be working correctly")

    sys.exit(0 if success else 1)
