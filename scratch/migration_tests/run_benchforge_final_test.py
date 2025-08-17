#!/usr/bin/env python3
"""Run final BenchForge test with all fixes applied."""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Run the test
sys.path.insert(0, "benchforge")
from flame.main_benchforge import main  # noqa: E402

# Set arguments
sys.argv = [
    "main_benchforge.py",
    "--mode",
    "inference",
    "--task",
    "fomc",
    "--model",
    "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "--max_tokens",
    "128",
    "--temperature",
    "0.0",
    "--batch_size",
    "50",  # Back to original batch size
    # Run full test with all 496 samples
]

print("Running full BenchForge FOMC test with all fixes...")
print("=" * 60)

try:
    main()
    print("\n" + "=" * 60)
    print("Test completed successfully!")

    # Check results
    import pandas as pd

    results_dir = Path("results/fomc")
    csv_files = list(results_dir.glob("*_together_ai_*.csv"))
    if csv_files:
        latest = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"\nAnalyzing results from: {latest.name}")

        df = pd.read_csv(latest)
        print(f"Total samples: {len(df)}")

        # Check extraction success
        if "extracted_labels" in df.columns:
            extracted = df["extracted_labels"].notna().sum()
            success_rate = (extracted / len(df)) * 100
            print(f"Extraction success: {extracted}/{len(df)} ({success_rate:.1f}%)")

            if success_rate >= 95:
                print("\n✅ SUCCESS: Achieved >95% extraction rate!")
            else:
                print(
                    f"\n⚠️  Current extraction rate: {success_rate:.1f}% (target: >95%)"
                )

            # Show sample of results
            print("\nFirst 5 extractions:")
            for i in range(min(5, len(df))):
                label = df["extracted_labels"].iloc[i]
                resp = (
                    df["llm_responses"].iloc[i][:50]
                    if pd.notna(df["llm_responses"].iloc[i])
                    else "None"
                )
                print(f"  {i + 1}. Response: '{resp}...' -> {label}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
