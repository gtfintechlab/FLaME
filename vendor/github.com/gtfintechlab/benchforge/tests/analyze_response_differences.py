#!/usr/bin/env python3
"""Analyze differences between BenchForge and FLAME responses."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd

# Load both CSV files
bf_df = pd.read_csv('numclaim_full_dataset_6788414b_benchforge.csv')
fl_df = pd.read_csv('numclaim_full_dataset_6788414b_flame.csv')

# Compare raw responses
identical = (bf_df['raw_responses'] == fl_df['raw_responses']).sum()
different = (bf_df['raw_responses'] != fl_df['raw_responses']).sum()
total = len(bf_df)

print("="*80)
print("RESPONSE COMPARISON ANALYSIS")
print("="*80)
print(f"\n📊 STATISTICS:")
print(f"Total samples: {total}")
print(f"Identical responses: {identical} ({identical/total*100:.2f}%)")
print(f"Different responses: {different} ({different/total*100:.2f}%)")

# Label agreement
label_agreement = (bf_df['extracted_labels'] == fl_df['extracted_labels']).sum()
print(f"\n🏷️ LABEL AGREEMENT:")
print(f"Same labels extracted: {label_agreement}/{total} ({label_agreement/total*100:.2f}%)")
print(f"Different labels: {total - label_agreement} ({(total - label_agreement)/total*100:.2f}%)")

# Find indices where responses differ
diff_indices = bf_df[bf_df['raw_responses'] != fl_df['raw_responses']].index

print(f"\n📝 SHOWING 15 EXAMPLES OF DIFFERENT RESPONSES:")
print("="*80)

for i, idx in enumerate(diff_indices[:15]):
    print(f"\n### DIFFERENCE {i+1} (Sample Index {idx}) ###")
    print(f"Input Sentence: \"{bf_df.loc[idx, 'sentences'][:200]}...\"")
    print(f"True Label: {bf_df.loc[idx, 'actual_labels']}")
    print("")
    
    print("BENCHFORGE Response:")
    bf_resp = bf_df.loc[idx, 'raw_responses'].strip()
    # Show first line and beginning of explanation
    lines = bf_resp.split('\n')
    print(f"  First line: {lines[0] if lines else 'N/A'}")
    if len(lines) > 1:
        print(f"  Explanation start: {lines[1][:150]}..." if len(lines[1]) > 150 else f"  Explanation: {lines[1]}")
    print(f"  → Extracted: {bf_df.loc[idx, 'extracted_labels']}, Correct: {bf_df.loc[idx, 'correct']}")
    
    print("")
    print("FLAME Response:")
    fl_resp = fl_df.loc[idx, 'raw_responses'].strip()
    lines = fl_resp.split('\n')
    print(f"  First line: {lines[0] if lines else 'N/A'}")
    if len(lines) > 1:
        print(f"  Explanation start: {lines[1][:150]}..." if len(lines[1]) > 150 else f"  Explanation: {lines[1]}")
    print(f"  → Extracted: {fl_df.loc[idx, 'extracted_labels']}, Correct: {fl_df.loc[idx, 'correct']}")
    
    # Check if they gave same label despite different text
    if bf_df.loc[idx, 'extracted_labels'] == fl_df.loc[idx, 'extracted_labels']:
        print("  ✅ Same label despite different explanations")
    else:
        print("  ❌ Different labels extracted")
    
    print("-"*80)

# Show some cases where labels differ
label_diff_indices = bf_df[bf_df['extracted_labels'] != fl_df['extracted_labels']].index
if len(label_diff_indices) > 0:
    print("\n🔍 CASES WHERE LABELS DIFFER:")
    print("="*80)
    for i, idx in enumerate(label_diff_indices[:5]):
        print(f"\n### LABEL DISAGREEMENT {i+1} (Index {idx}) ###")
        print(f"Sentence: \"{bf_df.loc[idx, 'sentences'][:150]}...\"")
        print(f"True Label: {bf_df.loc[idx, 'actual_labels']}")
        print(f"BenchForge: {bf_df.loc[idx, 'extracted_labels']} {'✅' if bf_df.loc[idx, 'correct'] else '❌'}")
        print(f"FLAME: {fl_df.loc[idx, 'extracted_labels']} {'✅' if fl_df.loc[idx, 'correct'] else '❌'}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Response text agreement: {identical/total*100:.2f}%")
print(f"✅ Label agreement: {label_agreement/total*100:.2f}%")
print(f"📊 BenchForge accuracy: {bf_df['correct'].sum()/total*100:.2f}%")
print(f"📊 FLAME accuracy: {fl_df['correct'].sum()/total*100:.2f}%")