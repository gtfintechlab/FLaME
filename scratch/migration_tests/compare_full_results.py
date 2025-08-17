#!/usr/bin/env python3
"""Compare full results from both implementations."""

import pandas as pd
from datetime import datetime

print("=" * 80)
print("FULL MIGRATION COMPARISON REPORT")
print("=" * 80)
print(f"Generated: {datetime.now()}")
print()

# Load results
native_path = "results/fomc/native_full_test_20250816_204342.csv"
benchforge_path = "results/fomc/fomc_together_ai_meta-llama_Llama-4-Scout-17B-16E-Instruct_16_08_2025_2051.csv"

print("Loading results...")
native_df = pd.read_csv(native_path)
benchforge_df = pd.read_csv(benchforge_path)

print(f"✅ Native FLAME: {len(native_df)} samples loaded")
print(f"✅ BenchForge: {len(benchforge_df)} samples loaded")
print()

# Column mapping
print("COLUMN MAPPING:")
print("-" * 40)
print(f"Native columns: {list(native_df.columns)[:5]}...")
print(f"BenchForge columns: {list(benchforge_df.columns)[:5]}...")
print()

# Map columns for comparison
# Native uses 'complete_responses' for extracted labels
native_labels = native_df.get("extracted_labels", native_df.get("complete_responses"))
# BenchForge uses 'extracted_response'
benchforge_labels = benchforge_df.get(
    "extracted_response", benchforge_df.get("extracted_labels")
)

if native_labels is None or benchforge_labels is None:
    print("❌ Cannot find extracted labels columns!")
    match_rate = 0
else:
    # Compare labels
    print("LABEL COMPARISON:")
    print("-" * 40)

    # Ensure same length for comparison
    min_len = min(len(native_labels), len(benchforge_labels))
    print(f"Comparing {min_len} samples...")

    # Calculate match rate
    matches = 0
    differences = []

    for i in range(min_len):
        if native_labels.iloc[i] == benchforge_labels.iloc[i]:
            matches += 1
        else:
            differences.append(
                {
                    "index": i,
                    "native": native_labels.iloc[i],
                    "benchforge": benchforge_labels.iloc[i],
                }
            )

    match_rate = matches / min_len * 100
    print(f"✅ Exact matches: {matches}/{min_len} ({match_rate:.2f}%)")

    if match_rate >= 95:
        print("🎉 IMPLEMENTATIONS ARE CONSISTENT (>95% match)!")
    else:
        print(f"⚠️ Found {len(differences)} differences")

    # Check extraction success rates
    native_success = native_labels.notna().sum()
    benchforge_success = benchforge_labels.notna().sum()

    print()
    print("EXTRACTION SUCCESS:")
    print("-" * 40)
    print(
        f"Native extraction success: {native_success}/{len(native_labels)} ({native_success / len(native_labels) * 100:.2f}%)"
    )
    print(
        f"BenchForge extraction success: {benchforge_success}/{len(benchforge_labels)} ({benchforge_success / len(benchforge_labels) * 100:.2f}%)"
    )

    # Compare accuracy if ground truth available
    if "actual_labels" in native_df and "ground_truth" in benchforge_df:
        print()
        print("ACCURACY COMPARISON:")
        print("-" * 40)

        native_correct = (native_labels == native_df["actual_labels"]).sum()
        benchforge_correct = (benchforge_labels == benchforge_df["ground_truth"]).sum()

        native_accuracy = native_correct / len(native_df) * 100
        benchforge_accuracy = benchforge_correct / len(benchforge_df) * 100

        print(
            f"Native accuracy: {native_correct}/{len(native_df)} ({native_accuracy:.2f}%)"
        )
        print(
            f"BenchForge accuracy: {benchforge_correct}/{len(benchforge_df)} ({benchforge_accuracy:.2f}%)"
        )

        if abs(native_accuracy - benchforge_accuracy) < 2:
            print("✅ Accuracy is comparable (within 2%)")
        else:
            print(
                f"⚠️ Accuracy difference: {abs(native_accuracy - benchforge_accuracy):.2f}%"
            )

    # Label distribution
    print()
    print("LABEL DISTRIBUTION:")
    print("-" * 40)

    native_dist = native_labels.value_counts()
    benchforge_dist = benchforge_labels.value_counts()

    print("Native distribution:")
    for label, count in native_dist.items():
        print(f"  {label}: {count} ({count / len(native_labels) * 100:.1f}%)")

    print("\nBenchForge distribution:")
    for label, count in benchforge_dist.items():
        print(f"  {label}: {count} ({count / len(benchforge_labels) * 100:.1f}%)")

    # Show sample differences
    if differences:
        print()
        print("SAMPLE DIFFERENCES (first 10):")
        print("-" * 40)
        for diff in differences[:10]:
            print(
                f"Sample {diff['index']}: native={diff['native']}, benchforge={diff['benchforge']}"
            )
        if len(differences) > 10:
            print(f"... and {len(differences) - 10} more differences")

# Performance timing (from logs)
print()
print("PERFORMANCE METRICS:")
print("-" * 40)
native_time = 83.82  # From native run log
benchforge_time = 430.0  # Approximate from timestamps (20:44:08 to 20:51:18)
print(f"Native FLAME time: {native_time:.2f} seconds")
print(f"BenchForge time: {benchforge_time:.2f} seconds")
print(f"Performance ratio: {benchforge_time / native_time:.2f}x")

if benchforge_time <= 2 * native_time:
    print("✅ Performance within 2x target")
else:
    print(f"⚠️ Performance is {benchforge_time / native_time:.2f}x slower (target: <2x)")

# Final assessment
print()
print("=" * 80)
print("MIGRATION READINESS ASSESSMENT")
print("=" * 80)

criteria_met = []
criteria_not_met = []

# Check criteria
if match_rate >= 95:
    criteria_met.append(f"Output consistency: {match_rate:.2f}% (target: >95%)")
else:
    criteria_not_met.append(f"Output consistency: {match_rate:.2f}% (target: >95%)")

if benchforge_time <= 2 * native_time:
    criteria_met.append(
        f"Performance: {benchforge_time / native_time:.2f}x (target: <2x)"
    )
else:
    criteria_not_met.append(
        f"Performance: {benchforge_time / native_time:.2f}x (target: <2x)"
    )

if benchforge_success / len(benchforge_labels) >= 0.95:
    criteria_met.append(
        f"Extraction success: {benchforge_success / len(benchforge_labels) * 100:.2f}% (target: >95%)"
    )
else:
    criteria_not_met.append(
        f"Extraction success: {benchforge_success / len(benchforge_labels) * 100:.2f}% (target: >95%)"
    )

if len(benchforge_df) == len(native_df):
    criteria_met.append("All samples processed successfully")
else:
    criteria_not_met.append(
        f"Sample count mismatch: {len(benchforge_df)} vs {len(native_df)}"
    )

print("✅ CRITERIA MET:")
for item in criteria_met:
    print(f"  - {item}")

if criteria_not_met:
    print("\n❌ CRITERIA NOT MET:")
    for item in criteria_not_met:
        print(f"  - {item}")

# Final recommendation
print()
print("=" * 80)
print("FINAL RECOMMENDATION:")
print("-" * 40)

if len(criteria_met) >= 3 and len(criteria_not_met) <= 1:
    print("🚀 BENCHFORGE IS READY FOR MIGRATION!")
    print("   The system meets most critical criteria.")
    print("   Safe to proceed with gradual production rollout.")
    if criteria_not_met:
        print(f"   Note: Address {criteria_not_met[0]} during optimization phase.")
else:
    print("⚠️ ADDITIONAL OPTIMIZATION NEEDED")
    print("   The following issues should be addressed:")
    for item in criteria_not_met:
        print(f"   - {item}")
    print("   Consider performance optimizations before full migration.")

print("=" * 80)
print(f"Report generated: {datetime.now()}")
print("=" * 80)
