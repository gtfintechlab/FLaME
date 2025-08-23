#!/usr/bin/env python3
"""Check if responses are identical between BenchForge and FLAME."""

import pandas as pd

# Load CSVs
bf_df = pd.read_csv("numclaim_full_dataset_3b3a24c2_benchforge.csv")
fl_df = pd.read_csv("numclaim_full_dataset_3b3a24c2_flame.csv")

print("=" * 60)
print("RESPONSE SIMILARITY CHECK")
print("=" * 60)

# Check if raw responses are identical
identical_responses = (bf_df["raw_responses"] == fl_df["raw_responses"]).sum()
total = len(bf_df)
print(
    f"\n📊 Identical Responses: {identical_responses}/{total} ({identical_responses / total * 100:.2f}%)"
)

# Sample some different responses
different = bf_df[bf_df["raw_responses"] != fl_df["raw_responses"]]
if len(different) > 0:
    print(f"\n🔍 Found {len(different)} different responses")
    print("\nFirst 3 differences:")
    for idx in different.index[:3]:
        print(f"\nSample {idx}:")
        print(f"BenchForge: {bf_df.loc[idx, 'raw_responses'][:100]}...")
        print(f"FLAME: {fl_df.loc[idx, 'raw_responses'][:100]}...")
else:
    print("\n⚠️ ALL RESPONSES ARE IDENTICAL!")
    print("This suggests caching is being used between runs.")

# Check API times
print("\n⏱️ API Times:")
print(f"BenchForge Total: {bf_df['api_times'].sum():.2f}s")
print(f"FLAME Total: {fl_df['api_times'].sum():.2f}s")
print(f"BenchForge Avg: {bf_df['api_times'].mean():.4f}s")
print(f"FLAME Avg: {fl_df['api_times'].mean():.4f}s")

# Check if FLAME times are suspiciously low (indicating cache hits)
if fl_df["api_times"].sum() < 5:
    print("\n🚨 FLAME times are suspiciously low - likely using cached responses!")

# Check a few responses to verify temperature=0
print("\n🌡️ Temperature Check (first 5 samples):")
for i in range(5):
    resp = bf_df.loc[i, "raw_responses"]
    if isinstance(resp, str) and len(resp) > 50:
        print(f"Sample {i}: {resp[:80]}...")
