#!/usr/bin/env python3
"""Verify FLAME evaluation compatibility with our improved outputs."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


def verify_flame_compatibility():
    """Test that FLAME evaluation works with our outputs."""

    print("=" * 70)
    print("VERIFYING FLAME EVALUATION COMPATIBILITY")
    print("=" * 70)

    # Load our most recent test output with fixed extraction
    test_file = "results/fomc/fomc_together_ai_meta-llama_Llama-4-Scout-17B-16E-Instruct_16_08_2025_2319.csv"

    print(f"\n1. Loading test file: {test_file}")
    df = pd.read_csv(test_file)

    print(f"   Loaded {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")

    # Check for FLAME-required columns
    print("\n2. Checking FLAME-required columns:")
    flame_columns = {
        "sentences": "Input text for display",
        "actual_labels": "Ground truth labels",
        "llm_responses": "Raw LLM responses",
        "extracted_labels": "Extracted predictions",
        "complete_responses": "Complete response objects for fallback",
    }

    for col, desc in flame_columns.items():
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"   ✓ {col}: {non_null}/{len(df)} non-null ({desc})")
        else:
            print(f"   ✗ {col}: MISSING!")

    # Check BenchForge columns too
    print("\n3. Checking BenchForge columns (aliases):")
    benchforge_columns = {
        "input": "BenchForge text field",
        "ground_truth": "BenchForge labels",
        "raw_response": "BenchForge raw responses",
        "extracted_response": "BenchForge extractions",
    }

    for col, desc in benchforge_columns.items():
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"   ✓ {col}: {non_null}/{len(df)} non-null ({desc})")
        else:
            print(f"   ✗ {col}: MISSING!")

    # Simulate FLAME evaluation
    print("\n4. Simulating FLAME evaluation:")
    print("-" * 40)

    # FLAME uses extracted_labels and actual_labels
    if "extracted_labels" in df.columns and "actual_labels" in df.columns:
        # Filter to non-null values
        valid_mask = df["extracted_labels"].notna() & df["actual_labels"].notna()
        valid_df = df[valid_mask].copy()

        print(f"   Valid samples for evaluation: {len(valid_df)}/{len(df)}")

        if len(valid_df) > 0:
            # Map labels to numeric values for sklearn
            label_map = {"DOVISH": 0, "HAWKISH": 1, "NEUTRAL": 2}

            # Convert string labels to numeric
            y_true = valid_df["actual_labels"].map(lambda x: label_map.get(x, x))
            y_pred = valid_df["extracted_labels"].map(lambda x: label_map.get(x, x))

            # Filter to samples where both mappings succeeded
            numeric_mask = (
                pd.to_numeric(y_true, errors="coerce").notna()
                & pd.to_numeric(y_pred, errors="coerce").notna()
            )
            y_true = y_true[numeric_mask]
            y_pred = y_pred[numeric_mask]

            if len(y_true) > 0:
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                f1_macro = f1_score(y_true, y_pred, average="macro")
                f1_weighted = f1_score(y_true, y_pred, average="weighted")
                precision = precision_score(y_true, y_pred, average="macro")
                recall = recall_score(y_true, y_pred, average="macro")

                print("\n   EVALUATION METRICS:")
                print(f"   Accuracy:        {accuracy:.4f}")
                print(f"   F1 (macro):      {f1_macro:.4f}")
                print(f"   F1 (weighted):   {f1_weighted:.4f}")
                print(f"   Precision:       {precision:.4f}")
                print(f"   Recall:          {recall:.4f}")

                # Classification report
                print("\n   CLASSIFICATION REPORT:")
                report = classification_report(
                    y_true,
                    y_pred,
                    target_names=["DOVISH", "HAWKISH", "NEUTRAL"],
                    digits=3,
                )
                for line in report.split("\n"):
                    if line:
                        print(f"   {line}")

    # Test fallback extraction from complete_responses
    print("\n5. Testing fallback extraction capability:")
    print("-" * 40)

    if "complete_responses" in df.columns:
        # Count how many have complete responses
        has_complete = df["complete_responses"].notna().sum()
        print(f"   Complete responses available: {has_complete}/{len(df)}")

        # For samples where primary extraction failed, check if we could extract from complete_responses
        failed_primary = df["extracted_labels"].isna()
        has_complete_fallback = df.loc[failed_primary, "complete_responses"].notna()

        print(f"   Failed primary extractions: {failed_primary.sum()}")
        print(f"   Have complete_responses for fallback: {has_complete_fallback.sum()}")

        if failed_primary.sum() > 0:
            print("\n   Fallback extraction could recover these failed samples")

    # Verify data quality
    print("\n6. Data Quality Verification:")
    print("-" * 40)

    # Check label distribution
    if "extracted_labels" in df.columns:
        label_dist = df["extracted_labels"].value_counts()
        print("   Extracted label distribution:")
        for label, count in label_dist.items():
            pct = count / len(df) * 100
            print(f"     {label}: {count} ({pct:.1f}%)")

    # Check for anomalies
    print("\n   Checking for anomalies:")

    # Empty responses
    if "llm_responses" in df.columns:
        empty_responses = (df["llm_responses"].str.strip() == "").sum()
        print(f"     Empty LLM responses: {empty_responses}")

    # Mismatched extractions
    if "extracted_labels" in df.columns and "extracted_response" in df.columns:
        mismatches = (df["extracted_labels"] != df["extracted_response"]).sum()
        if mismatches > 0:
            print(
                f"     ⚠️ Column mismatches (extracted_labels != extracted_response): {mismatches}"
            )
            # Show examples
            mismatch_df = df[df["extracted_labels"] != df["extracted_response"]].head(3)
            for idx, row in mismatch_df.iterrows():
                print(
                    f"       Row {idx}: extracted_labels='{row['extracted_labels']}', extracted_response='{row['extracted_response']}'"
                )

    print("\n" + "=" * 70)
    print("COMPATIBILITY SUMMARY:")
    print("-" * 40)

    all_flame_present = all(col in df.columns for col in flame_columns.keys())
    all_benchforge_present = all(col in df.columns for col in benchforge_columns.keys())

    if all_flame_present and all_benchforge_present:
        print("✅ FULL COMPATIBILITY: Both FLAME and BenchForge columns present")
        print("   - FLAME evaluation will work correctly")
        print("   - BenchForge evaluation will work correctly")
        print("   - Fallback extraction available via complete_responses")
    elif all_flame_present:
        print("✓ FLAME COMPATIBLE: All required FLAME columns present")
        print("⚠️ BenchForge columns missing or incomplete")
    elif all_benchforge_present:
        print("✓ BENCHFORGE COMPATIBLE: All BenchForge columns present")
        print("⚠️ FLAME columns missing or incomplete")
    else:
        print("⚠️ PARTIAL COMPATIBILITY: Some required columns missing")

    return df


if __name__ == "__main__":
    df = verify_flame_compatibility()

    # Final confidence assessment
    print("\n" + "=" * 70)
    print("CONFIDENCE ASSESSMENT:")
    print("-" * 40)

    if "extracted_labels" in df.columns:
        extraction_rate = df["extracted_labels"].notna().sum() / len(df) * 100

        if extraction_rate >= 95:
            print(f"✅ HIGH CONFIDENCE: {extraction_rate:.1f}% extraction success")
            print("   - Extraction methods working correctly")
            print("   - FLAME evaluation compatible")
            print("   - Ready for production use")
        else:
            print(f"⚠️ MEDIUM CONFIDENCE: {extraction_rate:.1f}% extraction success")
            print("   - Below 95% threshold")
            print("   - Further investigation needed")
