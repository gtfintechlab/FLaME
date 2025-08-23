#!/usr/bin/env python3
"""Quick comparison tool for NumClaim results."""

import pandas as pd

# Load both result files
bf_df = pd.read_csv("numclaim_full_dataset_3b3a24c2_benchforge.csv")
fl_df = pd.read_csv("numclaim_full_dataset_3b3a24c2_flame.csv")

print("=" * 60)
print("NUMCLAIM RESULTS COMPARISON")
print("=" * 60)

# Overall metrics
print("\n📊 OVERALL METRICS:")
print(f"BenchForge Accuracy: {bf_df['correct'].mean() * 100:.2f}%")
print(f"FLAME Accuracy: {fl_df['correct'].mean() * 100:.2f}%")

# Agreement analysis
agreements = (bf_df["extracted_labels"] == fl_df["extracted_labels"]).sum()
print(
    f"\nLabel Agreement: {agreements}/{len(bf_df)} ({agreements / len(bf_df) * 100:.2f}%)"
)

# Find disagreements
disagreements = bf_df[bf_df["extracted_labels"] != fl_df["extracted_labels"]]
print(f"\n🔍 DISAGREEMENTS: {len(disagreements)} samples")

if len(disagreements) > 0:
    print("\nFirst 5 disagreements:")
    for idx in disagreements.index[:5]:
        print(f"\nSample {idx}:")
        print(f"  Text: {bf_df.loc[idx, 'sentences'][:80]}...")
        print(f"  Actual: {bf_df.loc[idx, 'actual_labels']}")
        print(
            f"  BenchForge: {bf_df.loc[idx, 'extracted_labels']} {'✅' if bf_df.loc[idx, 'correct'] else '❌'}"
        )
        print(
            f"  FLAME: {fl_df.loc[idx, 'extracted_labels']} {'✅' if fl_df.loc[idx, 'correct'] else '❌'}"
        )

# Samples where both got wrong
both_wrong = bf_df[(~bf_df["correct"]) & (~fl_df["correct"])]
print(f"\n❌ Both Wrong: {len(both_wrong)} samples")

# Samples where BenchForge correct but FLAME wrong
bf_only_correct = bf_df[(bf_df["correct"]) & (~fl_df["correct"])]
print(f"✅ BenchForge Only Correct: {len(bf_only_correct)} samples")

# Samples where FLAME correct but BenchForge wrong
fl_only_correct = bf_df[(~bf_df["correct"]) & (fl_df["correct"])]
print(f"✅ FLAME Only Correct: {len(fl_only_correct)} samples")

print("\n💡 Use the CSV files directly for detailed manual inspection!")
