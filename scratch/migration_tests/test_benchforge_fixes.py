#!/usr/bin/env python3
"""Quick test to verify BenchForge fixes."""

import os
import sys
import time

# Set environment variables
os.environ["USE_BENCHFORGE_FOMC"] = "true"
os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "")


def test_benchforge_fixes():
    """Test BenchForge with a small sample to verify fixes."""

    print("=" * 60)
    print("Testing BenchForge Fixes")
    print("=" * 60)

    # Check API key
    if not os.environ.get("TOGETHER_API_KEY"):
        print("ERROR: TOGETHER_API_KEY not set")
        return False

    print("\nRunning BenchForge with 10 samples to test fixes...")

    # Run BenchForge inference with small sample
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
        "--max_samples",
        "10",
        "--use-benchforge",
    ]

    import subprocess

    start_time = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

    print(f"✅ Completed in {elapsed:.2f} seconds")

    # Parse output to check extraction rate
    output = result.stdout + result.stderr

    # Look for extraction success indicators
    if "Extraction success rate:" in output:
        # Extract the rate
        for line in output.split("\n"):
            if "Extraction success rate:" in line:
                print(f"   {line.strip()}")
                # Check if rate is good
                if "%" in line:
                    rate_str = line.split("(")[1].split("%")[0]
                    try:
                        rate = float(rate_str)
                        if rate >= 80:
                            print(f"   ✅ Extraction rate {rate}% is good!")
                        else:
                            print(f"   ⚠️ Extraction rate {rate}% is still low")
                    except (ValueError, AttributeError):
                        pass

    # Check performance
    if elapsed < 20:  # Should be fast for 10 samples
        print(f"   ✅ Performance is good ({elapsed:.2f}s for 10 samples)")
    else:
        print(f"   ⚠️ Performance may still need work ({elapsed:.2f}s for 10 samples)")

    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"- Execution: {'✅ Successful' if result.returncode == 0 else '❌ Failed'}")
    print(f"- Performance: {elapsed:.2f}s for 10 samples")
    print("- Expected for 496 samples: ~{:.0f}s".format(elapsed * 496 / 10))
    print("=" * 60)

    return result.returncode == 0


if __name__ == "__main__":
    success = test_benchforge_fixes()
    sys.exit(0 if success else 1)
